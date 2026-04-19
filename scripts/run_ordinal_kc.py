"""Day 5 Part A — Ordinal CapsNet + KC Loss (differentiable QWK).

Combined training objective:
    loss = OrdinalMarginLoss(head_lengths, y) + gamma * KCLoss(head_lengths, y)

gamma = 0 reproduces Day 3 (regression-test hook).

Usage:
    uv run python scripts/run_ordinal_kc.py                      # uses YAML
    uv run python scripts/run_ordinal_kc.py --gamma 0.3 --name-suffix gamma_0.3
    uv run python scripts/run_ordinal_kc.py --fold-only 0        # 1-fold sweep helper
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

from src.losses.kc_loss import KCLoss
from src.losses.ordinal_loss import (
    OrdinalMarginLoss,
    count_non_monotonic,
    ordinal_labels,
    predict_grade_from_heads,
)
from src.models.ordinal_capsnet import OrdinalCapsNet


# ---------------------------------------------------------------------------
# Setup
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
# Eval
# ---------------------------------------------------------------------------

def eval_with_metrics(model, loader, device, margin, kc, gamma) -> dict:
    model.eval()
    all_lengths, all_labels = [], []
    loss_sum, margin_sum, kc_sum, n = 0.0, 0.0, 0.0, 0
    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            L = model(X)["head_lengths"]
            l_m = margin(L, y)
            l_k = kc(L, y)
            total = l_m + gamma * l_k
            loss_sum += total.item() * X.size(0)
            margin_sum += l_m.item() * X.size(0)
            kc_sum += l_k.item() * X.size(0)
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

    return {
        "qwk": cohen_kappa_score(labels_np, preds, weights="quadratic"),
        "accuracy": accuracy_score(labels_np, preds),
        "macro_f1": f1_score(labels_np, preds, average="macro"),
        "confusion_matrix": cm,
        "loss": loss_sum / n,
        "margin_loss": margin_sum / n,
        "kc_loss": kc_sum / n,
        "per_head_acc": per_head_acc,
        "per_grade_acc": per_grade_acc,
        "non_monotonic_rate": count_non_monotonic(lengths).item() / len(labels),
    }


# ---------------------------------------------------------------------------
# Training (one fold)
# ---------------------------------------------------------------------------

def train_one_fold(
    model, tr_loader, va_loader, device, margin, kc, gamma,
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
            L = model(X)["head_lengths"]
            loss = margin(L, y) + gamma * kc(L, y)
            loss.backward()
            optim.step()

        train_m = eval_with_metrics(model, tr_loader, device, margin, kc, gamma)
        val_m = eval_with_metrics(model, va_loader, device, margin, kc, gamma)

        history.append({
            "epoch": epoch,
            "train_loss": train_m["loss"],
            "train_margin": train_m["margin_loss"], "train_kc": train_m["kc_loss"],
            "train_qwk": train_m["qwk"],
            "train_accuracy": train_m["accuracy"], "train_macro_f1": train_m["macro_f1"],
            "val_loss": val_m["loss"],
            "val_margin": val_m["margin_loss"], "val_kc": val_m["kc_loss"],
            "qwk": val_m["qwk"],
            "accuracy": val_m["accuracy"], "macro_f1": val_m["macro_f1"],
            "val_non_monotonic_rate": val_m["non_monotonic_rate"],
        })

        if val_m["qwk"] > best_qwk:
            best_qwk, best_metrics, best_ep = val_m["qwk"], val_m, epoch
            no_improve = 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            no_improve += 1

        if verbose and (epoch == 1 or epoch % log_every == 0 or no_improve >= patience):
            per_head = "[" + " ".join(f"{a:.3f}" for a in val_m["per_head_acc"]) + "]"
            print(f"    ep {epoch:3d}  "
                  f"train[qwk={train_m['qwk']:.4f} m={train_m['margin_loss']:.4f} kc={train_m['kc_loss']:.4f}]  "
                  f"val[qwk={val_m['qwk']:.4f} acc={val_m['accuracy']:.4f} heads={per_head}]"
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
# Plots (reuse pattern)
# ---------------------------------------------------------------------------

def plot_fold_histories(fold_histories, plots_dir: Path, name: str):
    plots_dir.mkdir(parents=True, exist_ok=True)
    safe = name.lower().replace(" ", "_").replace("(", "").replace(")", "")
    pairs = [
        ("train_loss", "val_loss", "loss", "Total Loss"),
        ("train_margin", "val_margin", "margin", "Margin Loss"),
        ("train_kc", "val_kc", "kc", "KC (1 - QWK) Loss"),
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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/ordinal_capsnet.yaml",
                   help="Reuse the Day 3 config; KC is added via --gamma")
    p.add_argument("--gamma", type=float, default=0.3, help="KC loss weight")
    p.add_argument("--name-suffix", type=str, default=None)
    p.add_argument("--fold-only", type=int, default=None,
                   help="Run only fold N (for sweep); skip holdout eval")
    p.add_argument("--no-wandb", action="store_true")
    p.add_argument("--epochs", type=int, default=None, help="Override max epochs")
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
    epochs = args.epochs if args.epochs else cfg["training"]["epochs"]
    weight_decay = cfg["training"].get("weight_decay", 1e-4)

    gamma = args.gamma
    use_wandb = (not args.no_wandb) and args.fold_only is None

    suffix = args.name_suffix or f"kc_gamma_{gamma:.2f}".replace(".", "p")
    name = f"ordinal-kc-{suffix}"
    plots_dir = Path("results/ordinal_kc") / suffix

    if args.fold_only is None:
        plots_dir.mkdir(parents=True, exist_ok=True)
    print(f"Device: {device}  gamma={gamma}  suffix={suffix}")

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
    print(f"CV pool: {len(features)}, Holdout: {len(X_holdout)}")

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    splits = list(skf.split(features, labels))

    if args.fold_only is not None:
        splits = [splits[args.fold_only]]
        print(f"FOLD-ONLY mode: fold {args.fold_only}")

    margin = OrdinalMarginLoss(num_classes=num_classes)
    kc = KCLoss(num_classes=num_classes)

    run = None
    if use_wandb:
        run = wandb.init(
            project=cfg["wandb"]["project"], group="day5-ordinal-kc", name=name,
            config={
                "model": "OrdinalCapsNet + KC Loss",
                **{k: v for k, v in cfg["model"].items()},
                "gamma": gamma,
                "lr": lr, "batch_size": batch_size, "epochs": epochs,
                "patience": patience, "n_folds": n_folds, "seed": seed,
                "weight_decay": weight_decay, "holdout_frac": holdout_frac,
            },
            reinit="finish_previous",
        )

    holdout_loader = None
    if args.fold_only is None:
        _, holdout_loader = make_loaders(X_holdout[:1], X_holdout, y_holdout[:1], y_holdout, batch_size)

    fold_metrics, fold_histories, fold_holdout = [], [], []
    sum_cm = np.zeros((num_classes, num_classes), dtype=np.int64)

    t_global = time.time()
    for fold_idx_rel, (tr_idx, va_idx) in enumerate(splits):
        fold_idx = args.fold_only if args.fold_only is not None else fold_idx_rel
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

        t0 = time.time()
        print(f"\nFold {fold_idx + 1}/{n_folds}:")
        metrics, history = train_one_fold(
            model, tr_loader, va_loader, device, margin, kc, gamma,
            lr=lr, max_epochs=epochs, patience=patience, weight_decay=weight_decay,
            verbose=True, log_every=max(1, epochs // 10),
        )
        sec_per_epoch = (time.time() - t0) / metrics["final_epoch"]

        holdout_m = None
        if holdout_loader is not None:
            holdout_m = eval_with_metrics(model, holdout_loader, device, margin, kc, gamma)
            fold_holdout.append(holdout_m)

        fold_metrics.append(metrics)
        fold_histories.append(history)
        sum_cm += metrics["confusion_matrix"]

        per_head_str = "[" + " ".join(f"{a:.3f}" for a in metrics["per_head_acc"]) + "]"
        per_grade_str = " ".join(f"{a:.3f}" for a in metrics["per_grade_acc"])
        if holdout_m is not None:
            print(f"  -> val QWK={metrics['qwk']:.4f} Acc={metrics['accuracy']:.4f} F1={metrics['macro_f1']:.4f} heads={per_head_str}")
            print(f"     per-grade: {per_grade_str}  (G0 G1 G2 G3 G4)")
            print(f"     holdout QWK={holdout_m['qwk']:.4f} Acc={holdout_m['accuracy']:.4f} "
                  f"F1={holdout_m['macro_f1']:.4f}  (best@{metrics['best_epoch']}, "
                  f"stopped@{metrics['final_epoch']}, {sec_per_epoch:.2f}s/ep)")
        else:
            print(f"  -> val QWK={metrics['qwk']:.4f} Acc={metrics['accuracy']:.4f} F1={metrics['macro_f1']:.4f}")
            print(f"     per-grade: {per_grade_str}  (G0 G1 G2 G3 G4)")

        if use_wandb:
            wandb.log({
                f"fold_{fold_idx+1}/val_qwk": metrics["qwk"],
                f"fold_{fold_idx+1}/val_accuracy": metrics["accuracy"],
                f"fold_{fold_idx+1}/val_macro_f1": metrics["macro_f1"],
                f"fold_{fold_idx+1}/holdout_qwk": holdout_m["qwk"] if holdout_m else None,
                f"fold_{fold_idx+1}/holdout_accuracy": holdout_m["accuracy"] if holdout_m else None,
                f"fold_{fold_idx+1}/best_epoch": metrics["best_epoch"],
                f"fold_{fold_idx+1}/sec_per_epoch": sec_per_epoch,
            })

    elapsed = time.time() - t_global

    # Aggregates (only meaningful if we ran all folds)
    qwks = [m["qwk"] for m in fold_metrics]
    accs = [m["accuracy"] for m in fold_metrics]
    f1s = [m["macro_f1"] for m in fold_metrics]
    per_head_matrix = np.array([m["per_head_acc"] for m in fold_metrics])
    per_grade_matrix = np.array([m["per_grade_acc"] for m in fold_metrics])

    result = {
        "gamma": gamma,
        "qwk_mean": float(np.mean(qwks)), "qwk_std": float(np.std(qwks)),
        "accuracy_mean": float(np.mean(accs)), "accuracy_std": float(np.std(accs)),
        "macro_f1_mean": float(np.mean(f1s)), "macro_f1_std": float(np.std(f1s)),
        "per_head_acc_mean": per_head_matrix.mean(axis=0).tolist(),
        "per_head_acc_std": per_head_matrix.std(axis=0).tolist(),
        "per_grade_acc_mean": per_grade_matrix.mean(axis=0).tolist(),
        "per_grade_acc_std": per_grade_matrix.std(axis=0).tolist(),
    }
    if fold_holdout:
        h_q = [m["qwk"] for m in fold_holdout]
        h_a = [m["accuracy"] for m in fold_holdout]
        h_f = [m["macro_f1"] for m in fold_holdout]
        result.update({
            "holdout_qwk_mean": float(np.mean(h_q)), "holdout_qwk_std": float(np.std(h_q)),
            "holdout_accuracy_mean": float(np.mean(h_a)), "holdout_accuracy_std": float(np.std(h_a)),
            "holdout_macro_f1_mean": float(np.mean(h_f)), "holdout_macro_f1_std": float(np.std(h_f)),
        })

    print()
    print("=" * 80)
    print(f"{name} — {len(splits)}-fold — {elapsed:.1f}s (gamma={gamma})")
    print("=" * 80)
    print(f"Val:     QWK={result['qwk_mean']:.4f}+/-{result['qwk_std']:.4f}  "
          f"Acc={result['accuracy_mean']:.4f}+/-{result['accuracy_std']:.4f}  "
          f"F1={result['macro_f1_mean']:.4f}+/-{result['macro_f1_std']:.4f}")
    if "holdout_qwk_mean" in result:
        print(f"Holdout: QWK={result['holdout_qwk_mean']:.4f}+/-{result['holdout_qwk_std']:.4f}  "
              f"Acc={result['holdout_accuracy_mean']:.4f}+/-{result['holdout_accuracy_std']:.4f}  "
              f"F1={result['holdout_macro_f1_mean']:.4f}+/-{result['holdout_macro_f1_std']:.4f}")
    grades = ["G0 (No DR)", "G1 (Mild)", "G2 (Moderate)", "G3 (Severe)", "G4 (PDR)"]
    print("\nPer-grade accuracy:")
    for i, g in enumerate(grades):
        print(f"  {g:<16s}  {result['per_grade_acc_mean'][i]:.4f} +/- {result['per_grade_acc_std'][i]:.4f}")
    print("=" * 80)

    if args.fold_only is None:
        plot_fold_histories(fold_histories, plots_dir, name)
        plot_confusion_matrix(sum_cm, plots_dir, name)
        (plots_dir / "summary.json").write_text(json.dumps({
            "model": name, "gamma": gamma, "result": result,
        }, indent=2))
        print(f"\nSaved plots and summary.json -> {plots_dir}/")

    if use_wandb and run is not None:
        wandb_summary = {
            "qwk_mean": result["qwk_mean"], "qwk_std": result["qwk_std"],
            "accuracy_mean": result["accuracy_mean"], "accuracy_std": result["accuracy_std"],
            "macro_f1_mean": result["macro_f1_mean"], "macro_f1_std": result["macro_f1_std"],
            "gamma": gamma,
            "grade_1_acc": result["per_grade_acc_mean"][1],
            "grade_3_acc": result["per_grade_acc_mean"][3],
        }
        if "holdout_qwk_mean" in result:
            wandb_summary.update({
                "holdout_qwk_mean": result["holdout_qwk_mean"],
                "holdout_accuracy_mean": result["holdout_accuracy_mean"],
                "holdout_macro_f1_mean": result["holdout_macro_f1_mean"],
            })
        wandb.summary.update(wandb_summary)
        for img in sorted(plots_dir.glob("*.png")):
            wandb.log({img.stem: wandb.Image(str(img))})
        run.finish()


if __name__ == "__main__":
    main()
