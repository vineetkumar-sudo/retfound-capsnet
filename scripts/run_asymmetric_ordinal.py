"""Day 4 — full 5-fold + holdout run of Ordinal CapsNet with asymmetric loss.

Near-copy of scripts/run_ordinal_capsnet.py. Only the loss changes
(AsymmetricOrdinalMarginLoss with configurable m_plus/m_minus/lambda_under/over).

Usage:
    uv run python scripts/run_asymmetric_ordinal.py                        # uses YAML
    uv run python scripts/run_asymmetric_ordinal.py --m-plus 0.95 --m-minus 0.05
    uv run python scripts/run_asymmetric_ordinal.py --lambda-under 2.0 --name-suffix C_mod_asym
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
import wandb
import yaml
from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from torch.utils.data import DataLoader, TensorDataset

from src.losses.asymmetric_loss import AsymmetricOrdinalMarginLoss, compute_niu_weights
from src.losses.ordinal_loss import (
    count_non_monotonic,
    ordinal_labels,
    predict_grade_from_heads,
)
from src.models.ordinal_capsnet import OrdinalCapsNet


# ---------------------------------------------------------------------------
# Setup helpers
# ---------------------------------------------------------------------------

def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def get_device() -> torch.device:
    if torch.cuda.is_available(): return torch.device("cuda")
    if torch.backends.mps.is_available(): return torch.device("mps")
    return torch.device("cpu")


def make_loaders(X_tr, X_va, y_tr, y_va, batch_size):
    tr = TensorDataset(torch.from_numpy(X_tr).float(), torch.from_numpy(y_tr).long())
    va = TensorDataset(torch.from_numpy(X_va).float(), torch.from_numpy(y_va).long())
    return (DataLoader(tr, batch_size=batch_size, shuffle=True),
            DataLoader(va, batch_size=batch_size, shuffle=False))


# ---------------------------------------------------------------------------
# Eval with per-head + per-grade accuracy + asymmetry ratio
# ---------------------------------------------------------------------------

def eval_ordinal(model, loader, device, loss_fn) -> dict:
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

    p_gt_k = lengths[:, :, 1]
    bin_preds = (p_gt_k > 0.5).long().numpy()
    bin_true = ordinal_labels(labels, 5).long().numpy()
    per_head_acc = [accuracy_score(bin_true[:, k], bin_preds[:, k]) for k in range(4)]

    cm = confusion_matrix(labels_np, preds, labels=range(5))
    row_sums = cm.sum(axis=1)
    per_grade_acc = [cm[i, i] / row_sums[i] if row_sums[i] > 0 else 0.0 for i in range(5)]
    below = sum(cm[i, j] for i in range(5) for j in range(5) if i > j)
    above = sum(cm[i, j] for i in range(5) for j in range(5) if i < j)
    asym_ratio = (below / above) if above > 0 else float("inf")

    return {
        "qwk": cohen_kappa_score(labels_np, preds, weights="quadratic"),
        "accuracy": accuracy_score(labels_np, preds),
        "macro_f1": f1_score(labels_np, preds, average="macro"),
        "confusion_matrix": cm,
        "loss": loss_sum / n,
        "per_head_acc": per_head_acc,
        "per_grade_acc": per_grade_acc,
        "asym_ratio_below_over_above": asym_ratio,
        "non_monotonic_rate": count_non_monotonic(lengths).item() / len(labels),
    }


# ---------------------------------------------------------------------------
# Training (one fold)
# ---------------------------------------------------------------------------

def train_one_fold(
    model, tr_loader, va_loader, device, loss_fn,
    lr, max_epochs, patience, weight_decay,
    verbose=False, log_every=10,
) -> tuple[dict, list[dict]]:
    model = model.to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=float(lr), weight_decay=float(weight_decay))

    best_qwk, best_metrics, best_ep, no_improve = -1.0, {}, 0, 0
    best_state = None
    history: list[dict] = []

    for epoch in range(1, max_epochs + 1):
        model.train()
        for X, y in tr_loader:
            X, y = X.to(device), y.to(device)
            optim.zero_grad()
            loss = loss_fn(model(X)["head_lengths"], y)
            loss.backward()
            optim.step()

        train_m = eval_ordinal(model, tr_loader, device, loss_fn)
        val_m = eval_ordinal(model, va_loader, device, loss_fn)

        history.append({
            "epoch": epoch,
            "train_loss": train_m["loss"], "train_qwk": train_m["qwk"],
            "train_accuracy": train_m["accuracy"], "train_macro_f1": train_m["macro_f1"],
            "val_loss": val_m["loss"], "qwk": val_m["qwk"],
            "accuracy": val_m["accuracy"], "macro_f1": val_m["macro_f1"],
            "val_non_monotonic_rate": val_m["non_monotonic_rate"],
            "val_asym_ratio": val_m["asym_ratio_below_over_above"],
        })

        if val_m["qwk"] > best_qwk:
            best_qwk, best_metrics, best_ep = val_m["qwk"], val_m, epoch
            no_improve = 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            no_improve += 1

        if verbose and (epoch == 1 or epoch % log_every == 0 or no_improve >= patience):
            per_head = "[" + " ".join(f"{a:.3f}" for a in val_m["per_head_acc"]) + "]"
            print(f"    ep {epoch:3d}  train[qwk={train_m['qwk']:.4f}]  "
                  f"val[qwk={val_m['qwk']:.4f} acc={val_m['accuracy']:.4f} "
                  f"U/O={val_m['asym_ratio_below_over_above']:.3f} heads={per_head}]"
                  + ("  *best*" if epoch == best_ep else ""))

        if no_improve >= patience:
            if verbose:
                print(f"    early stop @ ep {epoch} (best qwk={best_qwk:.4f} @ ep {best_ep})")
            break

    best_metrics["best_epoch"] = best_ep
    best_metrics["final_epoch"] = history[-1]["epoch"]
    if best_state is not None:
        model.load_state_dict(best_state)
    return best_metrics, history


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_fold_histories(fold_histories, plots_dir: Path, name: str):
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
            ax.text(0.02, 0.02, f"final train-val gap: {mean_tr[-1] - mean_va[-1]:+.4f}",
                    transform=ax.transAxes, fontsize=9,
                    bbox=dict(facecolor="white", alpha=0.7, edgecolor="gray"))
        ax.set_xlabel("Epoch"); ax.set_ylabel(ylabel)
        ax.set_title(f"{name} — {ylabel} (train vs val)")
        ax.legend(); ax.grid(True, alpha=0.3); fig.tight_layout()
        fig.savefig(plots_dir / f"{safe}_{fname}.png", dpi=150)
        plt.close(fig)


def plot_confusion_matrix(cm: np.ndarray, plots_dir: Path, name: str):
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


def plot_per_head_acc(per_head_mean, per_head_std, plots_dir: Path, name: str):
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
    p.add_argument("--config", default="configs/asymmetric_ordinal.yaml")
    p.add_argument("--m-plus", type=float, default=None)
    p.add_argument("--m-minus", type=float, default=None)
    p.add_argument("--lambda-under", type=float, default=None)
    p.add_argument("--lambda-over", type=float, default=None)
    p.add_argument("--use-niu", action="store_true")
    p.add_argument("--name-suffix", type=str, default=None,
                   help="Appended to wandb run name and output subdir")
    p.add_argument("--no-wandb", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    device = get_device()

    seed = cfg["data"]["seed"]
    feature_dim = cfg["data"]["feature_dim"]
    num_classes = cfg["data"]["num_classes"]
    n_folds = cfg["data"]["n_folds"]
    holdout_frac = cfg["data"].get("holdout_split", 0.10)
    patience = cfg["training"]["early_stopping_patience"]
    batch_size = cfg["training"]["batch_size"]
    lr = cfg["training"]["lr"]
    epochs = cfg["training"]["epochs"]
    weight_decay = cfg["training"].get("weight_decay", 1e-4)

    # Loss hyperparams — CLI overrides YAML
    lcfg = cfg.get("loss", {})
    m_plus = args.m_plus if args.m_plus is not None else lcfg.get("m_plus", 0.9)
    m_minus = args.m_minus if args.m_minus is not None else lcfg.get("m_minus", 0.1)
    lambda_base = lcfg.get("lambda_base", 0.5)
    lambda_under = args.lambda_under if args.lambda_under is not None else lcfg.get("lambda_under", 1.0)
    lambda_over = args.lambda_over if args.lambda_over is not None else lcfg.get("lambda_over", 1.0)
    use_niu = args.use_niu or lcfg.get("use_niu_weights", False)

    use_wandb = cfg["wandb"].get("enabled", True) and not args.no_wandb
    base_name = cfg["wandb"]["name"]
    name = f"{base_name}-{args.name_suffix}" if args.name_suffix else base_name
    plots_dir = Path(cfg["output"]["plots_dir"]) / (args.name_suffix or "default")
    plots_dir.mkdir(parents=True, exist_ok=True)

    print(f"Device: {device}")
    print(f"Loss config: m+={m_plus}, m-={m_minus}, "
          f"lambda_under={lambda_under}, lambda_over={lambda_over}, use_niu={use_niu}")
    print(f"Output: {plots_dir}/\n")

    # Data
    features_all = np.load(cfg["data"]["features_path"])
    labels_all = np.load(cfg["data"]["labels_path"])
    pool_idx, holdout_idx = train_test_split(
        np.arange(len(features_all)), test_size=holdout_frac,
        stratify=labels_all, random_state=seed,
    )
    features = features_all[pool_idx]
    labels = labels_all[pool_idx]
    X_holdout, y_holdout = features_all[holdout_idx], labels_all[holdout_idx]
    print(f"CV pool: {len(features)} samples  |  Holdout: {len(X_holdout)}")
    print(f"Label dist (pool): {np.bincount(labels)}\n")

    # Compute per-head weights once on the full pool (consistent across folds)
    per_head_weights = None
    if use_niu:
        per_head_weights = compute_niu_weights(labels, num_classes=num_classes)
        print(f"Niu per-head weights: {[f'{w:.3f}' for w in per_head_weights]}\n")

    def make_loss() -> AsymmetricOrdinalMarginLoss:
        return AsymmetricOrdinalMarginLoss(
            num_classes=num_classes,
            m_plus=m_plus, m_minus=m_minus, lambda_base=lambda_base,
            lambda_under=lambda_under, lambda_over=lambda_over,
            per_head_weights=per_head_weights,
        )

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    splits = list(skf.split(features, labels))

    run = None
    if use_wandb:
        run = wandb.init(
            project=cfg["wandb"]["project"], group=cfg["wandb"]["group"], name=name,
            config={
                "model": "OrdinalCapsNet + AsymmetricMargin",
                **{k: v for k, v in cfg["model"].items()},
                "m_plus": m_plus, "m_minus": m_minus,
                "lambda_base": lambda_base,
                "lambda_under": lambda_under, "lambda_over": lambda_over,
                "use_niu_weights": use_niu,
                "per_head_weights": per_head_weights,
                "lr": lr, "batch_size": batch_size, "epochs": epochs,
                "patience": patience, "n_folds": n_folds, "seed": seed,
                "weight_decay": weight_decay, "holdout_frac": holdout_frac,
            },
            reinit="finish_previous",
        )

    _, holdout_loader = make_loaders(X_holdout[:1], X_holdout, y_holdout[:1], y_holdout, batch_size)

    fold_metrics, fold_histories, fold_holdout = [], [], []
    sum_cm = np.zeros((num_classes, num_classes), dtype=np.int64)

    t_global = time.time()
    for fold_idx, (tr_idx, va_idx) in enumerate(splits):
        X_tr, X_va = features[tr_idx], features[va_idx]
        y_tr, y_va = labels[tr_idx], labels[va_idx]

        torch.manual_seed(seed + fold_idx)
        np.random.seed(seed + fold_idx)

        tr_loader, va_loader = make_loaders(X_tr, X_va, y_tr, y_va, batch_size)
        model = OrdinalCapsNet(
            feature_dim=feature_dim,
            num_primary=cfg["model"]["num_primary"],
            primary_dim=cfg["model"]["primary_dim"],
            num_classes=num_classes,
            caps_dim=cfg["model"]["caps_dim"],
            routing_iters=cfg["model"]["routing_iters"],
            dropout=cfg["model"].get("dropout", 0.0),
        )
        loss_fn = make_loss()

        t0 = time.time()
        print(f"Fold {fold_idx + 1}/{n_folds}:")
        metrics, history = train_one_fold(
            model, tr_loader, va_loader, device, loss_fn,
            lr=lr, max_epochs=epochs, patience=patience, weight_decay=weight_decay,
            verbose=True, log_every=max(1, epochs // 10),
        )
        sec_per_epoch = (time.time() - t0) / metrics["final_epoch"]

        holdout_m = eval_ordinal(model, holdout_loader, device, loss_fn)
        fold_holdout.append(holdout_m)
        fold_metrics.append(metrics)
        fold_histories.append(history)
        sum_cm += metrics["confusion_matrix"]

        per_head_str = "[" + " ".join(f"{a:.3f}" for a in metrics["per_head_acc"]) + "]"
        per_grade_str = " ".join(f"{a:.3f}" for a in metrics["per_grade_acc"])
        print(f"  -> val QWK={metrics['qwk']:.4f} Acc={metrics['accuracy']:.4f} "
              f"F1={metrics['macro_f1']:.4f} "
              f"U/O={metrics['asym_ratio_below_over_above']:.3f} heads={per_head_str}")
        print(f"     per-grade: {per_grade_str}  (G0 G1 G2 G3 G4)")
        print(f"     holdout QWK={holdout_m['qwk']:.4f} Acc={holdout_m['accuracy']:.4f} "
              f"F1={holdout_m['macro_f1']:.4f} "
              f"U/O={holdout_m['asym_ratio_below_over_above']:.3f}  "
              f"(best@{metrics['best_epoch']}, stopped@{metrics['final_epoch']}, {sec_per_epoch:.2f}s/ep)\n")

        if use_wandb:
            log = {
                f"fold_{fold_idx+1}/val_qwk": metrics["qwk"],
                f"fold_{fold_idx+1}/val_accuracy": metrics["accuracy"],
                f"fold_{fold_idx+1}/val_macro_f1": metrics["macro_f1"],
                f"fold_{fold_idx+1}/val_asym_ratio": metrics["asym_ratio_below_over_above"],
                f"fold_{fold_idx+1}/val_non_monotonic_rate": metrics["non_monotonic_rate"],
                f"fold_{fold_idx+1}/holdout_qwk": holdout_m["qwk"],
                f"fold_{fold_idx+1}/holdout_accuracy": holdout_m["accuracy"],
                f"fold_{fold_idx+1}/holdout_macro_f1": holdout_m["macro_f1"],
                f"fold_{fold_idx+1}/best_epoch": metrics["best_epoch"],
                f"fold_{fold_idx+1}/sec_per_epoch": sec_per_epoch,
            }
            for k, acc in enumerate(metrics["per_head_acc"]):
                log[f"fold_{fold_idx+1}/head_{k}_acc"] = acc
            for k, acc in enumerate(metrics["per_grade_acc"]):
                log[f"fold_{fold_idx+1}/grade_{k}_acc"] = acc
            wandb.log(log)

    # Aggregate
    qwks = [m["qwk"] for m in fold_metrics]
    accs = [m["accuracy"] for m in fold_metrics]
    f1s = [m["macro_f1"] for m in fold_metrics]
    nmr = [m["non_monotonic_rate"] for m in fold_metrics]
    uor = [m["asym_ratio_below_over_above"] for m in fold_metrics]
    per_head_matrix = np.array([m["per_head_acc"] for m in fold_metrics])
    per_grade_matrix = np.array([m["per_grade_acc"] for m in fold_metrics])

    h_q = [m["qwk"] for m in fold_holdout]
    h_a = [m["accuracy"] for m in fold_holdout]
    h_f = [m["macro_f1"] for m in fold_holdout]
    h_nmr = [m["non_monotonic_rate"] for m in fold_holdout]
    h_uor = [m["asym_ratio_below_over_above"] for m in fold_holdout]

    gaps_qwk = []
    for hist, m in zip(fold_histories, fold_metrics):
        row = hist[m["best_epoch"] - 1]
        gaps_qwk.append(row["train_qwk"] - row["qwk"])

    result = {
        "qwk_mean": float(np.mean(qwks)), "qwk_std": float(np.std(qwks)),
        "accuracy_mean": float(np.mean(accs)), "accuracy_std": float(np.std(accs)),
        "macro_f1_mean": float(np.mean(f1s)), "macro_f1_std": float(np.std(f1s)),
        "non_monotonic_rate_mean": float(np.mean(nmr)),
        "asym_ratio_mean": float(np.mean(uor)),
        "per_head_acc_mean": per_head_matrix.mean(axis=0).tolist(),
        "per_head_acc_std": per_head_matrix.std(axis=0).tolist(),
        "per_grade_acc_mean": per_grade_matrix.mean(axis=0).tolist(),
        "per_grade_acc_std": per_grade_matrix.std(axis=0).tolist(),
        "train_val_qwk_gap_mean": float(np.mean(gaps_qwk)),
        "holdout_qwk_mean": float(np.mean(h_q)), "holdout_qwk_std": float(np.std(h_q)),
        "holdout_accuracy_mean": float(np.mean(h_a)), "holdout_accuracy_std": float(np.std(h_a)),
        "holdout_macro_f1_mean": float(np.mean(h_f)), "holdout_macro_f1_std": float(np.std(h_f)),
        "holdout_non_monotonic_rate_mean": float(np.mean(h_nmr)),
        "holdout_asym_ratio_mean": float(np.mean(h_uor)),
    }

    elapsed = time.time() - t_global
    print("=" * 80)
    print(f"{name} — 5-fold CV — {elapsed:.1f}s")
    print("=" * 80)
    print(f"Val:     QWK={result['qwk_mean']:.4f}+/-{result['qwk_std']:.4f}  "
          f"Acc={result['accuracy_mean']:.4f}+/-{result['accuracy_std']:.4f}  "
          f"F1={result['macro_f1_mean']:.4f}+/-{result['macro_f1_std']:.4f}  "
          f"U/O={result['asym_ratio_mean']:.3f}")
    print(f"Holdout: QWK={result['holdout_qwk_mean']:.4f}+/-{result['holdout_qwk_std']:.4f}  "
          f"Acc={result['holdout_accuracy_mean']:.4f}+/-{result['holdout_accuracy_std']:.4f}  "
          f"F1={result['holdout_macro_f1_mean']:.4f}+/-{result['holdout_macro_f1_std']:.4f}  "
          f"U/O={result['holdout_asym_ratio_mean']:.3f}")
    print(f"Train-Val QWK gap (at best epoch, mean): {result['train_val_qwk_gap_mean']:+.4f}")
    print()
    grades = ["G0 (No DR)", "G1 (Mild)", "G2 (Moderate)", "G3 (Severe)", "G4 (PDR)"]
    print("Per-grade accuracy (5-fold mean +/- std):")
    for i, g in enumerate(grades):
        print(f"  {g:<16s}  {result['per_grade_acc_mean'][i]:.4f} +/- {result['per_grade_acc_std'][i]:.4f}")
    print()
    heads = ["y>0", "y>1", "y>2", "y>3"]
    print("Per-head binary accuracy:")
    for i, h in enumerate(heads):
        print(f"  {h:<6s}  {result['per_head_acc_mean'][i]:.4f} +/- {result['per_head_acc_std'][i]:.4f}")
    print("=" * 80)

    print(f"\nGenerating plots -> {plots_dir}/")
    plot_fold_histories(fold_histories, plots_dir, name)
    plot_confusion_matrix(sum_cm, plots_dir, name)
    plot_per_head_acc(result["per_head_acc_mean"], result["per_head_acc_std"], plots_dir, name)

    (plots_dir / "summary.json").write_text(json.dumps({
        "model": name,
        "loss": {
            "m_plus": m_plus, "m_minus": m_minus, "lambda_base": lambda_base,
            "lambda_under": lambda_under, "lambda_over": lambda_over,
            "use_niu": use_niu, "per_head_weights": per_head_weights,
        },
        "result": result,
    }, indent=2))

    if use_wandb and run is not None:
        wandb.log({
            "mean/val_qwk": result["qwk_mean"],
            "mean/val_accuracy": result["accuracy_mean"],
            "mean/val_macro_f1": result["macro_f1_mean"],
            "mean/holdout_qwk": result["holdout_qwk_mean"],
            "mean/holdout_accuracy": result["holdout_accuracy_mean"],
            "mean/asym_ratio": result["asym_ratio_mean"],
            "mean/train_val_qwk_gap": result["train_val_qwk_gap_mean"],
        })
        wandb.summary.update({
            "qwk_mean": result["qwk_mean"], "qwk_std": result["qwk_std"],
            "accuracy_mean": result["accuracy_mean"], "accuracy_std": result["accuracy_std"],
            "macro_f1_mean": result["macro_f1_mean"], "macro_f1_std": result["macro_f1_std"],
            "holdout_qwk_mean": result["holdout_qwk_mean"],
            "holdout_accuracy_mean": result["holdout_accuracy_mean"],
            "holdout_macro_f1_mean": result["holdout_macro_f1_mean"],
            "asym_ratio_mean": result["asym_ratio_mean"],
            "train_val_qwk_gap_mean": result["train_val_qwk_gap_mean"],
            "grade_1_acc_mean": result["per_grade_acc_mean"][1],
            "grade_3_acc_mean": result["per_grade_acc_mean"][3],
        })
        for img in sorted(plots_dir.glob("*.png")):
            wandb.log({img.stem: wandb.Image(str(img))})
        run.finish()

    print(f"\nDone. Results JSON at {plots_dir}/summary.json")


if __name__ == "__main__":
    main()
