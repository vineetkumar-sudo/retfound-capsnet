"""Day 3: Train the Ordinal CapsNet (K-1 binary heads) on cached RETFound features.

5-fold CV + frozen 10% holdout. AdamW + weight decay. Tracks train/val metrics
per epoch. Post-training: monotonicity and per-head binary accuracy reported.

Usage:
    uv run python scripts/run_ordinal_capsnet.py --demo       # fast sanity
    uv run python scripts/run_ordinal_capsnet.py              # full 5-fold
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, ".")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
import wandb
import yaml
from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from torch.utils.data import DataLoader, TensorDataset

from src.losses.ordinal_loss import (
    OrdinalMarginLoss,
    count_non_monotonic,
    ordinal_labels,
    predict_grade_from_heads,
)
from src.models.mlp_ordinal import MLPOrdinal
from src.models.ordinal_capsnet import OrdinalCapsNet


# ---------------------------------------------------------------------------
# Setup helpers
# ---------------------------------------------------------------------------

def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def make_loaders(X_tr, X_va, y_tr, y_va, batch_size: int):
    tr = TensorDataset(torch.from_numpy(X_tr).float(), torch.from_numpy(y_tr).long())
    va = TensorDataset(torch.from_numpy(X_va).float(), torch.from_numpy(y_va).long())
    return (DataLoader(tr, batch_size=batch_size, shuffle=True),
            DataLoader(va, batch_size=batch_size, shuffle=False))


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def collect_preds(model: OrdinalCapsNet, loader: DataLoader, device: torch.device) -> dict:
    """Run val/holdout loader, return per-sample arrays for Day 6 figures.

    Returns dict with numpy arrays:
      y_true (N,), y_pred (N,), head_probs (N, K-1), head_lengths (N, K-1, 2),
      routing_variance (N,)  -- per-sample Day 5B routing-agreement variance UQ
                               (mean std of coupling coeffs across routing iters,
                               averaged over heads / primaries / output caps).
    """
    model.eval()
    all_lengths, all_labels, all_rv = [], [], []
    for X, y in loader:
        X = X.to(device)
        out = model(X, return_routing_history=True)
        L = out["head_lengths"]           # (B, K-1, 2)
        rh = out["routing_history"]       # (H, R, B, P, 2)
        # Per-sample routing variance: same formula as src.uncertainty.routing_agreement_variance
        rv = rh.std(dim=1).mean(dim=(0, 2, 3))    # (B,)
        all_lengths.append(L.cpu())
        all_labels.append(y.cpu())
        all_rv.append(rv.cpu())
    lengths = torch.cat(all_lengths, dim=0)
    labels = torch.cat(all_labels, dim=0)
    rv_all = torch.cat(all_rv, dim=0)
    preds = predict_grade_from_heads(lengths).numpy().astype(np.int64)
    return {
        "y_true": labels.numpy().astype(np.int64),
        "y_pred": preds,
        "head_probs": lengths[:, :, 1].numpy().astype(np.float32),
        "head_lengths": lengths.numpy().astype(np.float32),
        "routing_variance": rv_all.numpy().astype(np.float32),
    }


def eval_ordinal(model: OrdinalCapsNet, loader: DataLoader, device: torch.device,
                 loss_fn: OrdinalMarginLoss) -> dict:
    """Single pass: metrics (QWK, acc, F1, CM), mean loss, monotonicity rate,
    per-head binary accuracy."""
    model.eval()
    all_lengths, all_labels, loss_sum, n = [], [], 0.0, 0
    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            out = model(X)
            L = out["head_lengths"]
            loss_sum += loss_fn(L, y).item() * X.size(0)
            n += X.size(0)
            all_lengths.append(L.cpu())
            all_labels.append(y.cpu())
    lengths = torch.cat(all_lengths, dim=0)
    labels = torch.cat(all_labels, dim=0)

    preds = predict_grade_from_heads(lengths).numpy()
    labels_np = labels.numpy().astype(int)

    # Per-head binary accuracy
    p_gt_k = lengths[:, :, 1]
    bin_preds = (p_gt_k > 0.5).long().numpy()
    bin_true = ordinal_labels(labels, 5).long().numpy()
    per_head_acc = [accuracy_score(bin_true[:, k], bin_preds[:, k]) for k in range(4)]

    return {
        "qwk": cohen_kappa_score(labels_np, preds, weights="quadratic"),
        "accuracy": accuracy_score(labels_np, preds),
        "macro_f1": f1_score(labels_np, preds, average="macro"),
        "confusion_matrix": confusion_matrix(labels_np, preds, labels=range(5)),
        "loss": loss_sum / n,
        "per_head_acc": per_head_acc,
        "non_monotonic_rate": count_non_monotonic(lengths).item() / len(labels),
    }


# ---------------------------------------------------------------------------
# Training (one fold)
# ---------------------------------------------------------------------------

def train_one_fold(
    model: OrdinalCapsNet, train_loader: DataLoader, val_loader: DataLoader,
    device: torch.device, lr: float, max_epochs: int, patience: int,
    num_classes: int, weight_decay: float = 1e-4,
    verbose: bool = False, log_every: int = 10,
) -> tuple[dict, list[dict], dict]:
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(lr), weight_decay=float(weight_decay))
    loss_fn = OrdinalMarginLoss(num_classes=num_classes)

    best_qwk, best_metrics, best_epoch, no_improve = -1.0, {}, 0, 0
    best_state: dict | None = None
    history: list[dict] = []

    for epoch in range(1, max_epochs + 1):
        # Train pass
        model.train()
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(X)
            loss = loss_fn(out["head_lengths"], y)
            loss.backward()
            optimizer.step()

        # Eval on train + val
        train_m = eval_ordinal(model, train_loader, device, loss_fn)
        val_m = eval_ordinal(model, val_loader, device, loss_fn)

        history.append({
            "epoch": epoch,
            "train_loss": train_m["loss"],
            "train_qwk": train_m["qwk"],
            "train_accuracy": train_m["accuracy"],
            "train_macro_f1": train_m["macro_f1"],
            "val_loss": val_m["loss"],
            "qwk": val_m["qwk"],
            "accuracy": val_m["accuracy"],
            "macro_f1": val_m["macro_f1"],
            "val_non_monotonic_rate": val_m["non_monotonic_rate"],
        })

        if val_m["qwk"] > best_qwk:
            best_qwk = val_m["qwk"]
            best_metrics = val_m
            best_epoch = epoch
            no_improve = 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            no_improve += 1

        if verbose and (epoch == 1 or epoch % log_every == 0 or no_improve >= patience):
            per_head = "[" + " ".join(f"{a:.3f}" for a in val_m["per_head_acc"]) + "]"
            print(f"    ep {epoch:3d}  train[qwk={train_m['qwk']:.4f} loss={train_m['loss']:.4f}]  "
                  f"val[qwk={val_m['qwk']:.4f} acc={val_m['accuracy']:.4f} "
                  f"non-mono={val_m['non_monotonic_rate'] * 100:.1f}% heads={per_head}]"
                  + ("  *best*" if epoch == best_epoch else ""))

        if no_improve >= patience:
            if verbose:
                print(f"    early stop @ ep {epoch} (best qwk={best_qwk:.4f} @ ep {best_epoch})")
            break

    best_metrics["best_epoch"] = best_epoch
    best_metrics["final_epoch"] = history[-1]["epoch"]
    if best_state is not None:
        model.load_state_dict(best_state)
    return best_metrics, history, best_state or {}


# ---------------------------------------------------------------------------
# Plots (same style as run_capsnet.py)
# ---------------------------------------------------------------------------

def plot_fold_histories(fold_histories: list[list[dict]], plots_dir: Path, name: str) -> None:
    plots_dir.mkdir(parents=True, exist_ok=True)
    safe = name.lower().replace(" ", "_").replace("(", "").replace(")", "")

    pairs = [
        ("train_loss", "val_loss", "loss", "Loss"),
        ("train_qwk", "qwk", "qwk", "QWK"),
        ("train_accuracy", "accuracy", "accuracy", "Accuracy"),
        ("train_macro_f1", "macro_f1", "macro_f1", "Macro F1"),
    ]
    for tr, va, fname, ylabel in pairs:
        fig, ax = plt.subplots(figsize=(8, 5))
        max_len = 0
        for fi, hist in enumerate(fold_histories):
            xs = [h["epoch"] for h in hist]
            ax.plot(xs, [h[tr] for h in hist], alpha=0.3, color="C0", linewidth=0.8,
                    label="train (folds)" if fi == 0 else None)
            ax.plot(xs, [h[va] for h in hist], alpha=0.3, color="C1", linewidth=0.8,
                    label="val (folds)" if fi == 0 else None)
            max_len = max(max_len, len(hist))

        mean_tr = [np.mean([h[e][tr] for h in fold_histories if e < len(h)]) for e in range(max_len)]
        mean_va = [np.mean([h[e][va] for h in fold_histories if e < len(h)]) for e in range(max_len)]
        ax.plot(range(1, len(mean_tr) + 1), mean_tr, color="C0", linewidth=2, label="train (mean)")
        ax.plot(range(1, len(mean_va) + 1), mean_va, color="C1", linewidth=2, label="val (mean)")

        if mean_tr and mean_va:
            gap = mean_tr[-1] - mean_va[-1]
            ax.text(0.02, 0.02, f"final train-val gap: {gap:+.4f}",
                    transform=ax.transAxes, fontsize=9,
                    bbox=dict(facecolor="white", alpha=0.7, edgecolor="gray"))

        ax.set_xlabel("Epoch"); ax.set_ylabel(ylabel)
        ax.set_title(f"{name} — {ylabel} (train vs val)")
        ax.legend(); ax.grid(True, alpha=0.3); fig.tight_layout()
        fig.savefig(plots_dir / f"{safe}_{fname}.png", dpi=150)
        plt.close(fig)

    # Non-monotonic rate plot
    fig, ax = plt.subplots(figsize=(8, 5))
    for fi, hist in enumerate(fold_histories):
        xs = [h["epoch"] for h in hist]
        ax.plot(xs, [h["val_non_monotonic_rate"] * 100 for h in hist],
                alpha=0.3, color="C3", linewidth=0.8,
                label="val (folds)" if fi == 0 else None)
    max_len = max(len(h) for h in fold_histories)
    mean_nm = [np.mean([h[e]["val_non_monotonic_rate"] for h in fold_histories if e < len(h)]) * 100
               for e in range(max_len)]
    ax.plot(range(1, len(mean_nm) + 1), mean_nm, color="C3", linewidth=2, label="val (mean)")
    ax.axhline(3.0, color="gray", linestyle="--", alpha=0.5, label="3% (spec: great)")
    ax.axhline(10.0, color="red", linestyle="--", alpha=0.5, label="10% (spec: warning)")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Non-monotonic predictions (%)")
    ax.set_title(f"{name} — Ordinal consistency over training")
    ax.legend(); ax.grid(True, alpha=0.3); fig.tight_layout()
    fig.savefig(plots_dir / f"{safe}_non_monotonic.png", dpi=150)
    plt.close(fig)


def plot_confusion_matrix(cm: np.ndarray, plots_dir: Path, name: str) -> None:
    class_names = [f"Grade {i}" for i in range(5)]
    cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm_pct, annot=True, fmt=".1f", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title(f"{name} (% per row, pooled over folds)")
    fig.tight_layout()
    fig.savefig(plots_dir / "confusion_matrix.png", dpi=150)
    plt.close(fig)


def plot_per_head_acc(per_head_mean: list[float], per_head_std: list[float],
                     plots_dir: Path, name: str) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    heads = ["y>0", "y>1", "y>2", "y>3"]
    x = np.arange(len(heads))
    ax.bar(x, per_head_mean, yerr=per_head_std, capsize=5,
           color=[f"C{i}" for i in range(4)], edgecolor="black")
    for i, (m, s) in enumerate(zip(per_head_mean, per_head_std)):
        ax.text(i, m + s + 0.005, f"{m:.3f}", ha="center", va="bottom",
                fontsize=10, fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(heads)
    ax.set_ylabel("Binary accuracy"); ax.set_ylim(0, 1.05)
    ax.set_title(f"{name} — Per-head binary accuracy (mean +/- std)")
    ax.grid(axis="y", alpha=0.3); fig.tight_layout()
    fig.savefig(plots_dir / "per_head_accuracy.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--demo", action="store_true")
    p.add_argument("--no-wandb", action="store_true")
    p.add_argument("--config", default="configs/ordinal_capsnet.yaml")
    p.add_argument("--seeds", default=None,
                   help="Comma-separated model-init seeds. If set, iterates over seeds; "
                        "outputs go to plots_dir/seed{S}/. If unset, uses cfg.data.seed (backwards-compatible).")
    return p.parse_args()


def parse_seeds(arg: str | None, default_seed: int) -> list[int]:
    if arg is None:
        return [default_seed]
    return [int(s) for s in arg.split(",") if s.strip()]


def build_model(cfg: dict, feature_dim: int, num_classes: int):
    """Dispatch on cfg.model.arch. Default is capsnet (backwards-compatible);
    set arch: mlp in the config to get the K-1 sigmoid MLP comparator."""
    mc = cfg["model"]
    arch = mc.get("arch", "capsnet")
    if arch == "capsnet":
        return OrdinalCapsNet(
            feature_dim=feature_dim,
            num_primary=mc["num_primary"],
            primary_dim=mc["primary_dim"],
            num_classes=num_classes,
            caps_dim=mc["caps_dim"],
            routing_iters=mc["routing_iters"],
            dropout=mc.get("dropout", 0.0),
            squash_variant=mc.get("squash_variant", "sabour"),
        )
    if arch == "mlp":
        return MLPOrdinal(
            feature_dim=feature_dim,
            hidden_dim=mc.get("hidden_dim", 256),
            num_classes=num_classes,
            num_hidden_layers=mc.get("num_hidden_layers", 2),
            dropout=mc.get("dropout", 0.0),
        )
    raise ValueError(f"Unknown model.arch='{arch}'; expected 'capsnet' or 'mlp'.")


def run_one_seed(
    cfg: dict, model_seed: int, device: torch.device,
    features, labels, X_holdout, y_holdout, splits,
    seed_plots_dir: Path, seed_suffix: str,
    name: str, use_wandb: bool, wandb_cfg: dict,
) -> dict:
    """Execute the fold loop for a single model-init seed.

    Fold splits + holdout are fixed (governed by cfg.data.seed); only model init RNG varies.
    Writes `seed_plots_dir/summary.json`, `seed_plots_dir/preds_fold{F}.npz` per fold, plus plots.
    Returns the per-seed result dict.
    """
    feature_dim = cfg["data"]["feature_dim"]
    num_classes = cfg["data"]["num_classes"]
    batch_size = cfg["training"]["batch_size"]
    lr = cfg["training"]["lr"]
    epochs = cfg["training"]["epochs"]
    patience = cfg["training"]["early_stopping_patience"]
    weight_decay = cfg["training"].get("weight_decay", 1e-4)

    seed_plots_dir.mkdir(parents=True, exist_ok=True)
    run_name = f"{name}{seed_suffix}"

    run = None
    if use_wandb:
        run = wandb.init(
            project=cfg["wandb"]["project"], group=cfg["wandb"]["group"], name=run_name,
            config={**wandb_cfg, "model_seed": model_seed},
            reinit="finish_previous",
        )

    holdout_loader = None
    if X_holdout is not None:
        _, holdout_loader = make_loaders(X_holdout[:1], X_holdout, y_holdout[:1], y_holdout, batch_size)

    fold_metrics, fold_histories, fold_holdout = [], [], []
    sum_cm = np.zeros((num_classes, num_classes), dtype=np.int64)

    t_global = time.time()
    for fold_idx, (tr_idx, va_idx) in enumerate(splits):
        X_tr, X_va = features[tr_idx], features[va_idx]
        y_tr, y_va = labels[tr_idx], labels[va_idx]

        torch.manual_seed(model_seed + fold_idx)
        np.random.seed(model_seed + fold_idx)

        tr_loader, va_loader = make_loaders(X_tr, X_va, y_tr, y_va, batch_size)
        model = build_model(cfg, feature_dim, num_classes)

        t0 = time.time()
        print(f"Fold {fold_idx + 1}/{len(splits)} (model_seed={model_seed}):")
        metrics, history, _ = train_one_fold(
            model, tr_loader, va_loader, device,
            lr=lr, max_epochs=epochs, patience=patience,
            num_classes=num_classes, weight_decay=weight_decay,
            verbose=True, log_every=max(1, epochs // 10),
        )
        sec_per_epoch = (time.time() - t0) / metrics["final_epoch"]

        # Capture per-fold predictions (best-state model) for Day 6 figures
        val_preds = collect_preds(model, va_loader, device)
        npz_payload = {
            "y_true": val_preds["y_true"],
            "y_pred": val_preds["y_pred"],
            "head_probs": val_preds["head_probs"],
            "head_lengths": val_preds["head_lengths"],
            "routing_variance": val_preds["routing_variance"],
        }

        holdout_m = None
        if holdout_loader is not None:
            loss_fn = OrdinalMarginLoss(num_classes=num_classes)
            holdout_m = eval_ordinal(model, holdout_loader, device, loss_fn)
            fold_holdout.append(holdout_m)
            hold_preds = collect_preds(model, holdout_loader, device)
            npz_payload.update({
                "holdout_y_true": hold_preds["y_true"],
                "holdout_y_pred": hold_preds["y_pred"],
                "holdout_head_probs": hold_preds["head_probs"],
                "holdout_head_lengths": hold_preds["head_lengths"],
                "holdout_routing_variance": hold_preds["routing_variance"],
            })

        np.savez_compressed(seed_plots_dir / f"preds_fold{fold_idx + 1}.npz", **npz_payload)

        fold_metrics.append(metrics)
        fold_histories.append(history)
        sum_cm += metrics["confusion_matrix"]

        head_str = "[" + " ".join(f"{a:.3f}" for a in metrics["per_head_acc"]) + "]"
        if holdout_m is not None:
            print(f"  -> val QWK={metrics['qwk']:.4f} Acc={metrics['accuracy']:.4f} F1={metrics['macro_f1']:.4f}  "
                  f"non-mono={metrics['non_monotonic_rate']*100:.2f}%  heads={head_str}")
            print(f"     holdout QWK={holdout_m['qwk']:.4f} Acc={holdout_m['accuracy']:.4f} F1={holdout_m['macro_f1']:.4f}  "
                  f"non-mono={holdout_m['non_monotonic_rate']*100:.2f}%  "
                  f"(best@{metrics['best_epoch']}, stopped@{metrics['final_epoch']}, {sec_per_epoch:.2f}s/ep)\n")
        else:
            print(f"  -> val QWK={metrics['qwk']:.4f} Acc={metrics['accuracy']:.4f} "
                  f"non-mono={metrics['non_monotonic_rate']*100:.2f}% "
                  f"(best@{metrics['best_epoch']}, stopped@{metrics['final_epoch']}, {sec_per_epoch:.2f}s/ep)\n")

        if use_wandb:
            log = {
                f"fold_{fold_idx+1}/val_qwk": metrics["qwk"],
                f"fold_{fold_idx+1}/val_accuracy": metrics["accuracy"],
                f"fold_{fold_idx+1}/val_macro_f1": metrics["macro_f1"],
                f"fold_{fold_idx+1}/val_non_monotonic_rate": metrics["non_monotonic_rate"],
                f"fold_{fold_idx+1}/best_epoch": metrics["best_epoch"],
                f"fold_{fold_idx+1}/sec_per_epoch": sec_per_epoch,
            }
            for k, acc in enumerate(metrics["per_head_acc"]):
                log[f"fold_{fold_idx+1}/head_{k}_acc"] = acc
            if holdout_m is not None:
                log[f"fold_{fold_idx+1}/holdout_qwk"] = holdout_m["qwk"]
                log[f"fold_{fold_idx+1}/holdout_accuracy"] = holdout_m["accuracy"]
                log[f"fold_{fold_idx+1}/holdout_macro_f1"] = holdout_m["macro_f1"]
                log[f"fold_{fold_idx+1}/holdout_non_monotonic_rate"] = holdout_m["non_monotonic_rate"]
            wandb.log(log)

    # Aggregate across folds (this seed)
    qwks = [m["qwk"] for m in fold_metrics]
    accs = [m["accuracy"] for m in fold_metrics]
    f1s = [m["macro_f1"] for m in fold_metrics]
    nmr = [m["non_monotonic_rate"] for m in fold_metrics]

    per_head_matrix = np.array([m["per_head_acc"] for m in fold_metrics])
    per_head_mean = per_head_matrix.mean(axis=0).tolist()
    per_head_std = per_head_matrix.std(axis=0).tolist()

    gaps_qwk = []
    for hist, m in zip(fold_histories, fold_metrics):
        row = hist[m["best_epoch"] - 1]
        gaps_qwk.append(row["train_qwk"] - row["qwk"])

    result = {
        "qwk_mean": float(np.mean(qwks)), "qwk_std": float(np.std(qwks)),
        "accuracy_mean": float(np.mean(accs)), "accuracy_std": float(np.std(accs)),
        "macro_f1_mean": float(np.mean(f1s)), "macro_f1_std": float(np.std(f1s)),
        "non_monotonic_rate_mean": float(np.mean(nmr)), "non_monotonic_rate_std": float(np.std(nmr)),
        "per_head_acc_mean": per_head_mean,
        "per_head_acc_std": per_head_std,
        "train_val_qwk_gap_mean": float(np.mean(gaps_qwk)),
        "per_fold_qwk": qwks,
    }
    if fold_holdout:
        h_q = [m["qwk"] for m in fold_holdout]
        h_a = [m["accuracy"] for m in fold_holdout]
        h_f = [m["macro_f1"] for m in fold_holdout]
        h_nmr = [m["non_monotonic_rate"] for m in fold_holdout]
        result.update({
            "holdout_qwk_mean": float(np.mean(h_q)), "holdout_qwk_std": float(np.std(h_q)),
            "holdout_accuracy_mean": float(np.mean(h_a)), "holdout_accuracy_std": float(np.std(h_a)),
            "holdout_macro_f1_mean": float(np.mean(h_f)), "holdout_macro_f1_std": float(np.std(h_f)),
            "holdout_non_monotonic_rate_mean": float(np.mean(h_nmr)),
            "per_fold_holdout_qwk": h_q,
        })

    elapsed = time.time() - t_global
    print("=" * 78)
    print(f"OrdinalCapsNet — seed={model_seed} — {len(splits)}-fold CV — total {elapsed:.1f}s")
    print("=" * 78)
    print(f"{'QWK':>16s}  {'Accuracy':>16s}  {'Macro F1':>16s}  {'Non-mono':>10s}")
    print(f"Val:     {result['qwk_mean']:.4f}+/-{result['qwk_std']:.4f}  "
          f"{result['accuracy_mean']:.4f}+/-{result['accuracy_std']:.4f}  "
          f"{result['macro_f1_mean']:.4f}+/-{result['macro_f1_std']:.4f}  "
          f"{result['non_monotonic_rate_mean']*100:.2f}%+/-{result['non_monotonic_rate_std']*100:.2f}%")
    if "holdout_qwk_mean" in result:
        print(f"Holdout: {result['holdout_qwk_mean']:.4f}+/-{result['holdout_qwk_std']:.4f}  "
              f"{result['holdout_accuracy_mean']:.4f}+/-{result['holdout_accuracy_std']:.4f}  "
              f"{result['holdout_macro_f1_mean']:.4f}+/-{result['holdout_macro_f1_std']:.4f}  "
              f"{result['holdout_non_monotonic_rate_mean']*100:.2f}%")
    print(f"Train-Val QWK gap (at best epoch, mean): {result['train_val_qwk_gap_mean']:+.4f}")
    print("=" * 78)

    # Per-seed plots
    print(f"\nGenerating plots -> {seed_plots_dir}/")
    plot_fold_histories(fold_histories, seed_plots_dir, name)
    plot_confusion_matrix(sum_cm, seed_plots_dir, name)
    plot_per_head_acc(per_head_mean, per_head_std, seed_plots_dir, name)

    # Persist per-fold histories for Day 6 training-curve figure.
    (seed_plots_dir / "fold_histories.json").write_text(json.dumps({
        "model_seed": model_seed,
        "fold_histories": fold_histories,
    }, indent=2))

    (seed_plots_dir / "summary.json").write_text(json.dumps({
        "model": name,
        "model_seed": model_seed,
        "config": {
            "num_primary": cfg["model"]["num_primary"],
            "primary_dim": cfg["model"]["primary_dim"],
            "caps_dim": cfg["model"]["caps_dim"],
            "routing_iters": cfg["model"]["routing_iters"],
            "dropout": cfg["model"].get("dropout", 0.0),
            "lr": lr, "batch_size": batch_size,
            "weight_decay": weight_decay,
        },
        "result": result,
        "per_fold": [
            {k: (v.tolist() if hasattr(v, "tolist") else v)
             for k, v in m.items() if k != "confusion_matrix"}
            for m in fold_metrics
        ],
    }, indent=2))

    if use_wandb and run is not None:
        heads = ["y>0", "y>1", "y>2", "y>3"]
        wsummary = {
            "qwk_mean": result["qwk_mean"], "qwk_std": result["qwk_std"],
            "accuracy_mean": result["accuracy_mean"], "accuracy_std": result["accuracy_std"],
            "macro_f1_mean": result["macro_f1_mean"], "macro_f1_std": result["macro_f1_std"],
            "non_monotonic_rate_mean": result["non_monotonic_rate_mean"],
            "train_val_qwk_gap_mean": result["train_val_qwk_gap_mean"],
            "model_seed": model_seed,
        }
        for i, _ in enumerate(heads):
            wsummary[f"head_{i}_acc_mean"] = per_head_mean[i]
        if "holdout_qwk_mean" in result:
            wsummary.update({
                "holdout_qwk_mean": result["holdout_qwk_mean"],
                "holdout_accuracy_mean": result["holdout_accuracy_mean"],
                "holdout_macro_f1_mean": result["holdout_macro_f1_mean"],
            })
        wandb.summary.update(wsummary)
        for img in sorted(seed_plots_dir.glob("*.png")):
            wandb.log({img.stem: wandb.Image(str(img))})
        run.finish()

    return result


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    device = get_device()

    data_seed = cfg["data"]["seed"]
    model_seeds = parse_seeds(args.seeds, data_seed)
    feature_dim = cfg["data"]["feature_dim"]
    num_classes = cfg["data"]["num_classes"]
    n_folds = cfg["data"]["n_folds"]
    holdout_frac = cfg["data"].get("holdout_split", 0.10)
    patience = cfg["training"]["early_stopping_patience"]
    batch_size = cfg["training"]["batch_size"]
    lr = cfg["training"]["lr"]
    epochs = cfg["training"]["epochs"]
    weight_decay = cfg["training"].get("weight_decay", 1e-4)
    use_wandb = cfg["wandb"].get("enabled", True) and not args.no_wandb
    plots_dir = Path(cfg["output"]["plots_dir"])

    if args.demo:
        d = cfg["demo"]
        n_folds = d["n_folds"]
        epochs = d["epochs"]
        cfg["training"]["epochs"] = epochs
        use_wandb = False
        plots_dir = plots_dir / "demo"
        print("=== DEMO MODE ===\n")

    plots_dir.mkdir(parents=True, exist_ok=True)
    print(f"Device: {device}")
    print(f"Config: num_classes={num_classes}, n_folds={n_folds}, "
          f"epochs<={epochs}, patience={patience}, weight_decay={weight_decay}, "
          f"dropout={cfg['model'].get('dropout', 0.0)}")
    print(f"Data seed: {data_seed}  |  Model seeds: {model_seeds}\n")

    # --- Load + split data (data_seed governs holdout + fold splits) ---
    features_all = np.load(cfg["data"]["features_path"])
    labels_all = np.load(cfg["data"]["labels_path"])
    if args.demo:
        rng = np.random.default_rng(data_seed)
        idx = rng.choice(len(features_all), size=cfg["demo"]["n_samples"], replace=False)
        features_all, labels_all = features_all[idx], labels_all[idx]
    print(f"Dataset: {features_all.shape[0]} samples, {features_all.shape[1]}-d features")
    print(f"Full label dist: {np.bincount(labels_all, minlength=num_classes)}")

    if not args.demo:
        pool_idx, holdout_idx = train_test_split(
            np.arange(len(features_all)), test_size=holdout_frac,
            stratify=labels_all, random_state=data_seed,
        )
        features = features_all[pool_idx]
        labels = labels_all[pool_idx]
        X_holdout, y_holdout = features_all[holdout_idx], labels_all[holdout_idx]
        print(f"CV pool: {len(features)} samples")
        print(f"Holdout: {len(X_holdout)} samples, dist={np.bincount(y_holdout)}\n")
    else:
        features, labels = features_all, labels_all
        X_holdout, y_holdout = None, None
        print()

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=data_seed) if n_folds > 1 else None
    if skf is None:
        tr_idx, va_idx = train_test_split(
            np.arange(len(features)), test_size=0.2, stratify=labels, random_state=data_seed,
        )
        splits = [(tr_idx, va_idx)]
    else:
        splits = list(skf.split(features, labels))

    name = cfg["wandb"]["name"]
    wandb_cfg = {
        "model": "OrdinalCapsNet (K-1 binary heads)",
        "feature_dim": feature_dim, "num_classes": num_classes,
        **{k: v for k, v in cfg["model"].items()},
        "lr": lr, "batch_size": batch_size, "epochs": epochs,
        "patience": patience, "n_folds": n_folds, "data_seed": data_seed,
        "weight_decay": weight_decay, "holdout_frac": holdout_frac,
    }

    # --- Run each model seed ---
    # Output layout:
    #   single seed == data_seed and no --seeds → plots_dir/ (backwards-compatible)
    #   otherwise → plots_dir/seed{S}/
    is_default_single = (args.seeds is None and len(model_seeds) == 1 and model_seeds[0] == data_seed)
    per_seed_results: list[dict] = []
    for model_seed in model_seeds:
        if is_default_single:
            seed_dir = plots_dir
            seed_suffix = ""
        else:
            seed_dir = plots_dir / f"seed{model_seed}"
            seed_suffix = f"-seed{model_seed}"

        result = run_one_seed(
            cfg, model_seed, device,
            features, labels, X_holdout, y_holdout, splits,
            seed_plots_dir=seed_dir, seed_suffix=seed_suffix,
            name=name, use_wandb=use_wandb, wandb_cfg=wandb_cfg,
        )
        per_seed_results.append({"model_seed": model_seed, "result": result})

    # --- Pooled summary across seeds (only when >1 seed) ---
    if len(model_seeds) > 1:
        pooled = {}
        for key in ("qwk", "accuracy", "macro_f1"):
            vals = [r["result"][f"{key}_mean"] for r in per_seed_results]
            pooled[f"{key}_mean"] = float(np.mean(vals))
            pooled[f"{key}_std_across_seeds"] = float(np.std(vals))
        if all("holdout_qwk_mean" in r["result"] for r in per_seed_results):
            for key in ("holdout_qwk", "holdout_accuracy", "holdout_macro_f1"):
                vals = [r["result"][f"{key}_mean"] for r in per_seed_results]
                pooled[f"{key}_mean"] = float(np.mean(vals))
                pooled[f"{key}_std_across_seeds"] = float(np.std(vals))
        (plots_dir / "summary_multiseed.json").write_text(json.dumps({
            "model": name,
            "model_seeds": model_seeds,
            "data_seed": data_seed,
            "per_seed": per_seed_results,
            "pooled_across_seeds": pooled,
        }, indent=2))
        print(f"\nDone. Multi-seed pooled summary -> {plots_dir}/summary_multiseed.json")
    else:
        print(f"\nDone. Results JSON at {plots_dir}/summary.json")


if __name__ == "__main__":
    main()
