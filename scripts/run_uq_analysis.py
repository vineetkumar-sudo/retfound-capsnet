"""Day 5 Part B — Uncertainty Quantification on the Day 3 Ordinal CapsNet.

Pipeline:
  1. Train Day 3 model on each of 5 folds (best-val-QWK checkpoint per fold).
  2. For each fold, compute 3 UQ methods on val set and on the frozen holdout.
  3. Pool across folds:
       (a) Box-plot: correct vs misclassified UQ (per method, with Mann-Whitney p).
       (b) Accuracy-coverage curves: reject-the-most-uncertain.
  4. Save publication-quality figures + summary JSON.

Usage:
    uv run python scripts/run_uq_analysis.py                   # full 5-fold + holdout
    uv run python scripts/run_uq_analysis.py --folds 2         # quick: 2 folds only
    uv run python scripts/run_uq_analysis.py --epochs 40       # quick training
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
import torch
from scipy.stats import mannwhitneyu
from sklearn.metrics import accuracy_score, cohen_kappa_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from torch.utils.data import DataLoader, TensorDataset

from src.losses.ordinal_loss import OrdinalMarginLoss, predict_grade_from_heads
from src.models.ordinal_capsnet import OrdinalCapsNet
from src.uncertainty import compute_all_uq


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_device() -> torch.device:
    if torch.cuda.is_available(): return torch.device("cuda")
    if torch.backends.mps.is_available(): return torch.device("mps")
    return torch.device("cpu")


def make_loaders(X_tr, X_va, y_tr, y_va, batch_size):
    tr = TensorDataset(torch.from_numpy(X_tr).float(), torch.from_numpy(y_tr).long())
    va = TensorDataset(torch.from_numpy(X_va).float(), torch.from_numpy(y_va).long())
    return (DataLoader(tr, batch_size=batch_size, shuffle=True),
            DataLoader(va, batch_size=batch_size, shuffle=False))


def train_day3_fold(X_tr, X_va, y_tr, y_va, device, epochs=100, patience=30,
                    lr=1e-3, weight_decay=1e-4, batch_size=64, verbose=False):
    tr_loader, va_loader = make_loaders(X_tr, X_va, y_tr, y_va, batch_size)
    model = OrdinalCapsNet(feature_dim=1024, num_classes=5, dropout=0.1).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    margin = OrdinalMarginLoss(num_classes=5)

    best_qwk, best_state, best_ep, no_improve = -1.0, None, 0, 0

    for epoch in range(1, epochs + 1):
        model.train()
        for X, y in tr_loader:
            X, y = X.to(device), y.to(device)
            optim.zero_grad()
            L = model(X)["head_lengths"]
            loss = margin(L, y)
            loss.backward()
            optim.step()

        # Val QWK
        model.eval()
        all_l, all_y = [], []
        with torch.no_grad():
            for X, y in va_loader:
                all_l.append(model(X.to(device))["head_lengths"].cpu())
                all_y.append(y)
        lengths = torch.cat(all_l, dim=0)
        labels = torch.cat(all_y, dim=0).numpy()
        preds = predict_grade_from_heads(lengths).numpy()
        qwk = cohen_kappa_score(labels, preds, weights="quadratic")

        if qwk > best_qwk:
            best_qwk, best_ep = qwk, epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        if verbose and (epoch == 1 or epoch % 20 == 0 or no_improve >= patience):
            print(f"    ep {epoch:3d}  val_qwk={qwk:.4f}"
                  + ("  *best*" if epoch == best_ep else ""))
        if no_improve >= patience:
            break

    model.load_state_dict(best_state)
    return model, best_qwk, best_ep


def gather_uq_and_preds(model, X: np.ndarray, y: np.ndarray, device: torch.device,
                        batch_size: int = 128) -> dict:
    """Run model with routing history, stack UQ across batches."""
    chunks_entropy, chunks_rv, chunks_margin, chunks_preds = [], [], [], []
    for i in range(0, len(X), batch_size):
        Xb = torch.from_numpy(X[i:i+batch_size]).float()
        out = compute_all_uq(model, Xb, device)
        chunks_entropy.append(out["digit_entropy"])
        chunks_rv.append(out["routing_variance"])
        chunks_margin.append(out["prediction_margin"])
        chunks_preds.append(predict_grade_from_heads(out["head_lengths"]))

    preds = torch.cat(chunks_preds, dim=0).numpy().astype(int)
    correct = (preds == y.astype(int))
    return {
        "digit_entropy": torch.cat(chunks_entropy, dim=0).numpy(),
        "routing_variance": torch.cat(chunks_rv, dim=0).numpy(),
        "prediction_margin": torch.cat(chunks_margin, dim=0).numpy(),
        "preds": preds,
        "labels": y.astype(int),
        "correct": correct,
    }


# ---------------------------------------------------------------------------
# Analysis / plots
# ---------------------------------------------------------------------------

def mann_whitney(pooled: dict, method: str) -> dict:
    u_correct = pooled[method][pooled["correct"]]
    u_wrong = pooled[method][~pooled["correct"]]
    stat, p = mannwhitneyu(u_wrong, u_correct, alternative="greater")
    return {
        "n_correct": int(u_correct.size),
        "n_wrong": int(u_wrong.size),
        "mean_correct": float(u_correct.mean()),
        "mean_wrong": float(u_wrong.mean()),
        "mw_statistic": float(stat),
        "p_value": float(p),
    }


def plot_box(pooled: dict, method: str, out_path: Path, pretty: str) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    u_correct = pooled[method][pooled["correct"]]
    u_wrong = pooled[method][~pooled["correct"]]
    bp = ax.boxplot(
        [u_correct, u_wrong],
        labels=[f"correct\n(n={u_correct.size})", f"misclassified\n(n={u_wrong.size})"],
        patch_artist=True, widths=0.5, showfliers=False,
    )
    for patch, color in zip(bp["boxes"], ["#7fb3d5", "#e8a87c"]):
        patch.set_facecolor(color); patch.set_edgecolor("black")
    for med in bp["medians"]:
        med.set_color("black")

    stat, p = mannwhitneyu(u_wrong, u_correct, alternative="greater")
    ax.set_ylabel(pretty)
    ax.set_title(f"{pretty}: correct vs misclassified\nMann-Whitney U p = {p:.3e}")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def accuracy_coverage_curve(pooled: dict, method: str) -> tuple[np.ndarray, np.ndarray]:
    u = pooled[method]
    c = pooled["correct"]
    # Ascending order of uncertainty -> keep the most confident first
    order = np.argsort(u)
    c_sorted = c[order]
    # Coverage fractions from 100% down; cumulative correctness ratio
    N = len(c_sorted)
    k = np.arange(1, N + 1)
    cum_correct = np.cumsum(c_sorted)
    accuracy_at_k = cum_correct / k
    coverage = k / N
    return coverage, accuracy_at_k


def plot_accuracy_coverage(pooled: dict, out_path: Path, title: str) -> None:
    methods = [
        ("digit_entropy", "DigitCap Entropy", "C0"),
        ("routing_variance", "Routing Variance", "C1"),
        ("prediction_margin", "Prediction Margin", "C2"),
    ]
    fig, ax = plt.subplots(figsize=(8, 6))
    for method, pretty, color in methods:
        cov, acc = accuracy_coverage_curve(pooled, method)
        ax.plot(cov, acc, color=color, linewidth=2, label=pretty)
    baseline_acc = float(pooled["correct"].mean())
    ax.axhline(baseline_acc, color="gray", linestyle="--", linewidth=1,
               label=f"baseline acc = {baseline_acc:.3f} (all samples)")
    ax.set_xlabel("Coverage (fraction of samples kept, most confident first)")
    ax.set_ylabel("Accuracy on kept samples")
    ax.set_title(title)
    ax.set_xlim(0, 1); ax.set_ylim(max(0, baseline_acc - 0.05), 1.01)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower left")
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--patience", type=int, default=30)
    ap.add_argument("--out", type=str, default="results/uq")
    args = ap.parse_args()

    device = get_device()
    print(f"Device: {device}")

    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)

    # Data + holdout split
    features_all = np.load("data/aptos/features/train_features.npy")
    labels_all = np.load("data/aptos/features/train_labels.npy")
    pool_idx, hold_idx = train_test_split(
        np.arange(len(features_all)), test_size=0.10,
        stratify=labels_all, random_state=42,
    )
    features = features_all[pool_idx]; labels = labels_all[pool_idx]
    X_hold = features_all[hold_idx]; y_hold = labels_all[hold_idx]
    print(f"CV pool: {len(features)}  Holdout: {len(X_hold)}\n")

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    splits = list(skf.split(features, labels))[:args.folds]

    # Accumulators — pool across all 5 folds
    val_pool = {"digit_entropy": [], "routing_variance": [], "prediction_margin": [],
                "correct": [], "preds": [], "labels": []}
    hold_pool = {k: [] for k in val_pool}
    per_fold_qwk: list[dict] = []

    t_global = time.time()
    for fold_idx, (tr_idx, va_idx) in enumerate(splits):
        X_tr, X_va = features[tr_idx], features[va_idx]
        y_tr, y_va = labels[tr_idx], labels[va_idx]
        torch.manual_seed(42 + fold_idx); np.random.seed(42 + fold_idx)

        print(f"--- Fold {fold_idx + 1}/{args.folds} — training Day 3 ---")
        t0 = time.time()
        model, best_qwk, best_ep = train_day3_fold(
            X_tr, X_va, y_tr, y_va, device,
            epochs=args.epochs, patience=args.patience, verbose=True,
        )
        print(f"  best val QWK={best_qwk:.4f} @ ep {best_ep}  ({time.time()-t0:.1f}s)")

        # UQ on val
        uq_val = gather_uq_and_preds(model, X_va, y_va, device)
        # UQ on holdout
        uq_hold = gather_uq_and_preds(model, X_hold, y_hold, device)

        val_qwk = cohen_kappa_score(uq_val["labels"], uq_val["preds"], weights="quadratic")
        hold_qwk = cohen_kappa_score(uq_hold["labels"], uq_hold["preds"], weights="quadratic")
        val_acc = accuracy_score(uq_val["labels"], uq_val["preds"])
        hold_acc = accuracy_score(uq_hold["labels"], uq_hold["preds"])
        per_fold_qwk.append({
            "fold": fold_idx + 1,
            "val_qwk": val_qwk, "val_acc": val_acc,
            "holdout_qwk": hold_qwk, "holdout_acc": hold_acc,
        })
        print(f"  UQ computed: val acc={val_acc:.4f}  holdout acc={hold_acc:.4f}\n")

        for k in val_pool:
            val_pool[k].append(uq_val[k])
            hold_pool[k].append(uq_hold[k])

    # Concatenate into pooled arrays
    for key in val_pool:
        val_pool[key] = np.concatenate(val_pool[key], axis=0)
        hold_pool[key] = np.concatenate(hold_pool[key], axis=0)

    elapsed = time.time() - t_global
    print(f"Total: {elapsed:.1f}s ({args.folds} folds)\n")

    # --- Mann-Whitney U test per UQ method on both val and holdout ---
    methods = ["digit_entropy", "routing_variance", "prediction_margin"]
    pretty_names = {
        "digit_entropy": "DigitCap Entropy",
        "routing_variance": "Routing Variance",
        "prediction_margin": "Prediction Margin",
    }

    stats: dict[str, dict] = {"val": {}, "holdout": {}}
    print("=" * 78)
    print("UQ validity: Mann-Whitney U test (H1: misclassified have HIGHER uncertainty)")
    print("=" * 78)
    print(f"{'method':<22s} {'split':<8s} {'mean_correct':>14s} {'mean_wrong':>12s} {'p-value':>12s}")
    print("-" * 78)
    for split_name, pool in [("val", val_pool), ("holdout", hold_pool)]:
        for m in methods:
            r = mann_whitney(pool, m)
            stats[split_name][m] = r
            mark = " ***" if r["p_value"] < 1e-3 else " **" if r["p_value"] < 0.01 else " *" if r["p_value"] < 0.05 else ""
            print(f"{pretty_names[m]:<22s} {split_name:<8s} "
                  f"{r['mean_correct']:>14.4f} {r['mean_wrong']:>12.4f} "
                  f"{r['p_value']:>12.3e}{mark}")
    print("-" * 78)
    print("*** p<0.001, ** p<0.01, * p<0.05")
    print()

    # --- Plots ---
    print(f"Saving plots -> {out_dir}/")
    for m in methods:
        plot_box(val_pool, m, out_dir / f"box_val_{m}.png", pretty_names[m])
        plot_box(hold_pool, m, out_dir / f"box_holdout_{m}.png", pretty_names[m])
    plot_accuracy_coverage(val_pool, out_dir / "acc_coverage_val.png",
                           "Accuracy-Coverage (val, 5-fold pooled)")
    plot_accuracy_coverage(hold_pool, out_dir / "acc_coverage_holdout.png",
                           "Accuracy-Coverage (holdout, 5-fold pooled)")

    # --- Coverage summary (what does acc at 80%/50% coverage look like?) ---
    print("\nAccuracy-coverage summary:")
    print(f"{'method':<22s} {'split':<8s} {'cov=1.0':>8s} {'cov=0.8':>8s} {'cov=0.5':>8s}")
    print("-" * 60)
    cov_summary: dict = {}
    for split_name, pool in [("val", val_pool), ("holdout", hold_pool)]:
        for m in methods:
            cov, acc = accuracy_coverage_curve(pool, m)
            # Values at the closest-to-target coverage indices
            def acc_at(target):
                idx = int(round(target * len(cov))) - 1
                return float(acc[max(0, min(idx, len(acc) - 1))])
            summary = {"cov_1.0": acc_at(1.0), "cov_0.8": acc_at(0.8), "cov_0.5": acc_at(0.5)}
            cov_summary.setdefault(split_name, {})[m] = summary
            print(f"{pretty_names[m]:<22s} {split_name:<8s} "
                  f"{summary['cov_1.0']:.4f}  {summary['cov_0.8']:.4f}  {summary['cov_0.5']:.4f}")

    # --- Save summary JSON ---
    summary = {
        "per_fold": per_fold_qwk,
        "mann_whitney": stats,
        "accuracy_coverage": cov_summary,
        "runtime_sec": elapsed,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nSaved summary.json and 8 plots to {out_dir}/")


if __name__ == "__main__":
    main()
