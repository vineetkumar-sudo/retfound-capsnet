"""Day 4 — single-fold sweep over 7 asymmetric-loss configurations.

Runs each config on fold 0 (of the stratified 5-fold split on the CV pool) to
convergence with early stopping, then prints a sorted table. Saves per-config
metrics + sweep_summary.json. Top-2 configs by val QWK (tie-break: Grade 1 acc)
are printed for easy promotion to a full 5-fold run.

Usage:
    uv run python scripts/sweep_asymmetric.py
    uv run python scripts/sweep_asymmetric.py --limit 1   # smoke test: config A only
    uv run python scripts/sweep_asymmetric.py --epochs 40 # quick mode
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, ".")

import numpy as np
import torch
import torch.nn as nn
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
from src.utils import enable_tf32


# ---------------------------------------------------------------------------
# Unit test — runs before sweep
# ---------------------------------------------------------------------------

def unit_test_asymmetry() -> None:
    """Verify that under-grading loss is lambda_under x over-grading loss
    when capsule-length magnitudes are identical and opposite in direction.
    """
    print("Running unit test: asymmetric direction weighting...")
    torch.manual_seed(0)

    # --- Under-grading case ---
    # Truth y=3 -> binary labels [1,1,1,0].
    # For head 2 (y>2), truth=1 but model outputs (neg=0.95, pos=0.05).
    # We isolate head 2 by keeping other heads at margin boundaries (no loss).
    under_lengths = torch.zeros(1, 4, 2)
    under_lengths[0, 0] = torch.tensor([0.05, 0.95])  # head 0, truth=1, correct
    under_lengths[0, 1] = torch.tensor([0.05, 0.95])  # head 1, truth=1, correct
    under_lengths[0, 2] = torch.tensor([0.95, 0.05])  # head 2, truth=1, UNDER-graded
    under_lengths[0, 3] = torch.tensor([0.95, 0.05])  # head 3, truth=0, correct
    y_under = torch.tensor([3])

    # --- Over-grading case ---
    # Truth y=1 -> binary labels [1,0,0,0].
    # For head 2 (y>2), truth=0 but model outputs (neg=0.05, pos=0.95).
    over_lengths = torch.zeros(1, 4, 2)
    over_lengths[0, 0] = torch.tensor([0.05, 0.95])  # head 0, truth=1, correct
    over_lengths[0, 1] = torch.tensor([0.95, 0.05])  # head 1, truth=0, correct
    over_lengths[0, 2] = torch.tensor([0.05, 0.95])  # head 2, truth=0, OVER-graded
    over_lengths[0, 3] = torch.tensor([0.95, 0.05])  # head 3, truth=0, correct
    y_over = torch.tensor([1])

    loss_fn = AsymmetricOrdinalMarginLoss(
        num_classes=5, m_plus=0.9, m_minus=0.1, lambda_base=0.5,
        lambda_under=2.0, lambda_over=1.0,
    )
    l_under = loss_fn(under_lengths, y_under).item()
    l_over = loss_fn(over_lengths, y_over).item()
    ratio = l_under / l_over
    print(f"  under-grading loss = {l_under:.6f}")
    print(f"  over-grading loss  = {l_over:.6f}")
    print(f"  ratio (expected 2.0) = {ratio:.6f}")
    assert abs(ratio - 2.0) < 1e-4, f"Asymmetric loss ratio wrong: {ratio}"
    print("  PASS\n")


# ---------------------------------------------------------------------------
# Training helpers (minimal; single fold only)
# ---------------------------------------------------------------------------

def get_device() -> torch.device:
    if torch.cuda.is_available():
        enable_tf32()
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def make_loaders(X_tr, X_va, y_tr, y_va, batch_size):
    tr = TensorDataset(torch.from_numpy(X_tr).float(), torch.from_numpy(y_tr).long())
    va = TensorDataset(torch.from_numpy(X_va).float(), torch.from_numpy(y_va).long())
    return (DataLoader(tr, batch_size=batch_size, shuffle=True),
            DataLoader(va, batch_size=batch_size, shuffle=False))


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
    return {
        "qwk": cohen_kappa_score(labels_np, preds, weights="quadratic"),
        "accuracy": accuracy_score(labels_np, preds),
        "macro_f1": f1_score(labels_np, preds, average="macro"),
        "confusion_matrix": cm,
        "loss": loss_sum / n,
        "per_head_acc": per_head_acc,
        "non_monotonic_rate": count_non_monotonic(lengths).item() / len(labels),
    }


def train_fold(X_tr, X_va, y_tr, y_va, loss_fn, device,
               batch_size=64, lr=1e-3, max_epochs=100, patience=30, weight_decay=1e-4,
               dropout=0.1) -> tuple[dict, int]:
    tr_loader, va_loader = make_loaders(X_tr, X_va, y_tr, y_va, batch_size)
    model = OrdinalCapsNet(feature_dim=1024, num_classes=5, dropout=dropout).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_qwk, best_metrics, best_ep, no_improve = -1.0, {}, 0, 0
    best_state = None

    for epoch in range(1, max_epochs + 1):
        model.train()
        for X, y in tr_loader:
            X, y = X.to(device), y.to(device)
            optim.zero_grad()
            loss = loss_fn(model(X)["head_lengths"], y)
            loss.backward()
            optim.step()
        m = eval_ordinal(model, va_loader, device, loss_fn)
        if m["qwk"] > best_qwk:
            best_qwk, best_metrics, best_ep = m["qwk"], m, epoch
            no_improve = 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            no_improve += 1
        if no_improve >= patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return best_metrics, best_ep


# ---------------------------------------------------------------------------
# Config definitions
# ---------------------------------------------------------------------------

CONFIGS = [
    dict(name="A_baseline",         m_plus=0.90, m_minus=0.10, lambda_under=1.0, lambda_over=1.0, use_niu=False),
    dict(name="B_mild_asym",        m_plus=0.90, m_minus=0.10, lambda_under=1.5, lambda_over=1.0, use_niu=False),
    dict(name="C_mod_asym",         m_plus=0.90, m_minus=0.10, lambda_under=2.0, lambda_over=1.0, use_niu=False),
    dict(name="D_strong_asym",      m_plus=0.90, m_minus=0.10, lambda_under=3.0, lambda_over=1.0, use_niu=False),
    dict(name="E_stricter_margin",  m_plus=0.95, m_minus=0.05, lambda_under=1.0, lambda_over=1.0, use_niu=False),
    dict(name="F_margin_weight",    m_plus=0.95, m_minus=0.05, lambda_under=2.0, lambda_over=1.0, use_niu=False),
    dict(name="G_niu",              m_plus=0.90, m_minus=0.10, lambda_under=1.0, lambda_over=1.0, use_niu=True),
]


# ---------------------------------------------------------------------------
# Main sweep
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None,
                    help="Only run the first N configs (smoke test)")
    ap.add_argument("--epochs", type=int, default=100,
                    help="Max epochs per config")
    ap.add_argument("--patience", type=int, default=30)
    ap.add_argument("--out", type=str, default="results/asymmetric/sweep_summary.json")
    args = ap.parse_args()

    unit_test_asymmetry()

    device = get_device()
    print(f"Device: {device}")

    # Load data + same holdout split as ordinal_capsnet
    features_all = np.load("data/aptos/features/train_features.npy")
    labels_all = np.load("data/aptos/features/train_labels.npy")
    pool_idx, holdout_idx = train_test_split(
        np.arange(len(features_all)), test_size=0.10, stratify=labels_all, random_state=42,
    )
    features = features_all[pool_idx]
    labels = labels_all[pool_idx]
    print(f"CV pool: {len(features)} samples, dist={np.bincount(labels)}\n")

    # Fold 0 of the stratified 5-fold
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    tr_idx, va_idx = next(iter(skf.split(features, labels)))
    X_tr, X_va = features[tr_idx], features[va_idx]
    y_tr, y_va = labels[tr_idx], labels[va_idx]
    print(f"Fold 0 — train: {len(X_tr)}  val: {len(X_va)}\n")

    configs = CONFIGS[: args.limit] if args.limit else CONFIGS

    results: list[dict] = []
    t_global = time.time()

    for i, cfg in enumerate(configs):
        name = cfg["name"]
        print(f"[{i+1}/{len(configs)}] {name}  "
              f"m+={cfg['m_plus']}  m-={cfg['m_minus']}  "
              f"λu={cfg['lambda_under']}  λo={cfg['lambda_over']}  niu={cfg['use_niu']}")

        per_head_weights = None
        if cfg["use_niu"]:
            per_head_weights = compute_niu_weights(y_tr, num_classes=5)
            print(f"       niu weights = {[f'{w:.3f}' for w in per_head_weights]}")

        loss_fn = AsymmetricOrdinalMarginLoss(
            num_classes=5,
            m_plus=cfg["m_plus"], m_minus=cfg["m_minus"], lambda_base=0.5,
            lambda_under=cfg["lambda_under"], lambda_over=cfg["lambda_over"],
            per_head_weights=per_head_weights,
        )

        # Reproducibility across configs
        torch.manual_seed(42)
        np.random.seed(42)

        t0 = time.time()
        m, best_ep = train_fold(
            X_tr, X_va, y_tr, y_va, loss_fn, device,
            max_epochs=args.epochs, patience=args.patience,
        )
        dt = time.time() - t0

        # Per-grade accuracy = normalized CM diagonal
        cm = m["confusion_matrix"]
        row_sums = cm.sum(axis=1)
        per_grade_acc = [cm[i, i] / row_sums[i] if row_sums[i] > 0 else 0.0 for i in range(5)]

        # Asymmetry ratio: under-grading errors / over-grading errors
        # Below-diagonal (i > j) = predictions lower than truth = under-grading
        below = sum(cm[i, j] for i in range(5) for j in range(5) if i > j)
        above = sum(cm[i, j] for i in range(5) for j in range(5) if i < j)
        asym_ratio = below / above if above > 0 else float("inf")

        result = {
            "name": name,
            **{k: v for k, v in cfg.items() if k != "name"},
            "val_qwk": m["qwk"],
            "val_accuracy": m["accuracy"],
            "val_macro_f1": m["macro_f1"],
            "val_non_monotonic_rate": m["non_monotonic_rate"],
            "per_head_acc": m["per_head_acc"],
            "per_grade_acc": per_grade_acc,
            "asym_ratio_below_over_above": asym_ratio,
            "best_epoch": best_ep,
            "runtime_sec": dt,
        }
        results.append(result)

        print(f"    val QWK={m['qwk']:.4f}  Acc={m['accuracy']:.4f}  F1={m['macro_f1']:.4f}  "
              f"non-mono={m['non_monotonic_rate']*100:.2f}%")
        print(f"    per-grade acc: {[f'{x:.3f}' for x in per_grade_acc]}  "
              f"(G0 G1 G2 G3 G4)")
        print(f"    under/over error ratio = {asym_ratio:.3f}  "
              f"(best@{best_ep}, {dt:.1f}s)\n")

    # Sort & report
    print("=" * 88)
    print("SWEEP RESULTS (sorted by val QWK desc, tie-break: Grade 1 accuracy)")
    print("=" * 88)
    results_sorted = sorted(
        results,
        key=lambda r: (r["val_qwk"], r["per_grade_acc"][1]),
        reverse=True,
    )
    hdr = (f"{'config':<20s} {'QWK':>7s} {'Acc':>7s} {'F1':>7s} "
           f"{'G0':>6s} {'G1':>6s} {'G2':>6s} {'G3':>6s} {'G4':>6s} "
           f"{'U/O':>6s} {'non-mono':>9s} {'best':>5s}")
    print(hdr)
    print("-" * len(hdr))
    for r in results_sorted:
        print(f"{r['name']:<20s} "
              f"{r['val_qwk']:.4f}  {r['val_accuracy']:.4f}  {r['val_macro_f1']:.4f}  "
              + "  ".join(f"{r['per_grade_acc'][g]:.3f}" for g in range(5)) + "  "
              f"{r['asym_ratio_below_over_above']:.3f}  "
              f"{r['val_non_monotonic_rate']*100:5.2f}%   "
              f"{r['best_epoch']:3d}")
    print()

    top2 = results_sorted[:2]
    print(f"TOP 2 — promote to full 5-fold:")
    for r in top2:
        print(f"  {r['name']}:  uv run python scripts/run_asymmetric_ordinal.py "
              f"--m-plus {r['m_plus']} --m-minus {r['m_minus']} "
              f"--lambda-under {r['lambda_under']} --lambda-over {r['lambda_over']} "
              + ("--use-niu " if r["use_niu"] else "")
              + f"--name-suffix {r['name']}")

    # Save
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({
            "total_runtime_sec": time.time() - t_global,
            "configs": [
                {**r, "confusion_matrix": None}  # drop ndarray
                for r in results
            ],
            "top_2": [r["name"] for r in top2],
        }, f, indent=2, default=lambda o: o.item() if hasattr(o, "item") else o)
    print(f"\nSweep summary saved to {out_path}")


if __name__ == "__main__":
    main()
