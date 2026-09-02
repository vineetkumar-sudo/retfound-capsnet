"""Day 8 — Cross-dataset evaluation: train Ordinal CapsNet on APTOS, test on IDRiD.

Protocol:
  * Trains 5 Ordinal CapsNet models on APTOS's 5-fold CV splits (seed=42) to
    produce an ensemble that matches the Day 3 champion. Epochs = 100, patience
    = 30 on val QWK (APTOS held-out fold), AdamW, weight_decay = 1e-4 — exactly
    the Day 3 hyperparameters.
  * For each fold's best model, predicts on the IDRiD 103-image test set, then
    averages P(y > k) across the 5 folds and decodes to a grade with the
    standard threshold-0.5 rule.

This is one-way cross-dataset: APTOS-trained weights, IDRiD test labels. It is
an out-of-distribution check for the learned feature→grade mapping on top of
frozen RETFound, not a re-training or fine-tune.

Outputs:
    results/cross_dataset/aptos_to_idrid/
        preds_fold{N}.npz          (per-fold IDRiD-test preds)
        summary.json               (ensembled metrics + per-fold)

Usage:
    uv run python scripts/cross_dataset_aptos_to_idrid.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, ".")

import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold, train_test_split
from torch.utils.data import DataLoader, TensorDataset

from src.evaluate import compute_all_metrics
from src.losses.ordinal_loss import OrdinalMarginLoss, predict_grade_from_heads
from src.models.ordinal_capsnet import OrdinalCapsNet
from src.utils import enable_tf32


# Day 3 champion hyperparameters (paper config)
NUM_CLASSES = 5
FEATURE_DIM = 1024
N_FOLDS = 5
SEED = 42
HOLDOUT_FRAC = 0.10
EPOCHS = 100
PATIENCE = 30
BATCH_SIZE = 64
LR = 1e-3
WEIGHT_DECAY = 1e-4


def get_device() -> torch.device:
    if torch.cuda.is_available():
        enable_tf32()
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def make_loader(X, y, batch_size: int, shuffle: bool) -> DataLoader:
    ds = TensorDataset(torch.from_numpy(X).float(),
                       torch.from_numpy(y).long())
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


def build_model() -> OrdinalCapsNet:
    return OrdinalCapsNet(
        feature_dim=FEATURE_DIM, num_classes=NUM_CLASSES,
        num_primary=32, primary_dim=8, caps_dim=16, routing_iters=3,
        dropout=0.1,
    )


@torch.no_grad()
def eval_qwk(model: OrdinalCapsNet, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    ys, ps = [], []
    for X, y in loader:
        X = X.to(device)
        L = model(X)["head_lengths"]
        ps.append(predict_grade_from_heads(L).cpu().numpy())
        ys.append(y.numpy())
    return compute_all_metrics(np.concatenate(ys), np.concatenate(ps))["qwk"]


@torch.no_grad()
def collect_head_probs(model: OrdinalCapsNet, loader: DataLoader,
                       device: torch.device) -> np.ndarray:
    """Return (N, K-1) P(y > k) per sample."""
    model.eval()
    chunks = []
    for X, _ in loader:
        X = X.to(device)
        L = model(X)["head_lengths"]
        chunks.append(L[:, :, 1].cpu().numpy())
    return np.concatenate(chunks, axis=0).astype(np.float32)


def train_one_fold(
    X_tr, y_tr, X_va, y_va, device: torch.device, fold_idx: int,
) -> OrdinalCapsNet:
    """Train on fold-tr, early-stop on fold-va, return best-val-QWK model."""
    torch.manual_seed(SEED + fold_idx)
    np.random.seed(SEED + fold_idx)

    tr_loader = make_loader(X_tr, y_tr, BATCH_SIZE, shuffle=True)
    va_loader = make_loader(X_va, y_va, BATCH_SIZE, shuffle=False)

    model = build_model().to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    loss_fn = OrdinalMarginLoss(num_classes=NUM_CLASSES)

    best_qwk, best_state, no_improve, best_epoch = -1.0, None, 0, 0
    for epoch in range(1, EPOCHS + 1):
        model.train()
        for X, y in tr_loader:
            X, y = X.to(device), y.to(device)
            optim.zero_grad()
            out = model(X)
            loss_fn(out["head_lengths"], y).backward()
            optim.step()
        q = eval_qwk(model, va_loader, device)
        if q > best_qwk:
            best_qwk = q
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch
            no_improve = 0
        else:
            no_improve += 1
        if no_improve >= PATIENCE:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    print(f"    fold {fold_idx + 1}: best val QWK={best_qwk:.4f} @ ep{best_epoch}")
    return model


def main() -> None:
    out_dir = Path("results/cross_dataset/aptos_to_idrid")
    out_dir.mkdir(parents=True, exist_ok=True)
    device = get_device()
    print(f"Device: {device}")

    # --- APTOS train (ex-holdout) ---
    X_aptos = np.load("data/aptos/features/train_features.npy")
    y_aptos = np.load("data/aptos/features/train_labels.npy")
    pool_idx, _ = train_test_split(
        np.arange(len(X_aptos)), test_size=HOLDOUT_FRAC,
        stratify=y_aptos, random_state=SEED,
    )
    X_pool, y_pool = X_aptos[pool_idx], y_aptos[pool_idx]
    print(f"APTOS CV pool: {X_pool.shape[0]} samples")

    # --- IDRiD test ---
    X_idrid = np.load("data/idrid/features/test_features.npy")
    y_idrid = np.load("data/idrid/features/test_labels.npy")
    print(f"IDRiD test:    {X_idrid.shape[0]} samples, "
          f"dist={np.bincount(y_idrid, minlength=5).tolist()}")

    idrid_loader = make_loader(X_idrid, y_idrid, BATCH_SIZE, shuffle=False)

    # --- 5-fold training on APTOS; per-fold eval on IDRiD test ---
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    per_fold_probs: list[np.ndarray] = []
    per_fold_metrics: list[dict] = []
    t0 = time.time()
    for fold_idx, (tr_idx, va_idx) in enumerate(skf.split(X_pool, y_pool)):
        print(f"\n=== APTOS fold {fold_idx + 1}/{N_FOLDS} ===")
        model = train_one_fold(
            X_pool[tr_idx], y_pool[tr_idx],
            X_pool[va_idx], y_pool[va_idx],
            device, fold_idx,
        )
        # Single-model IDRiD test eval
        probs = collect_head_probs(model, idrid_loader, device)  # (N, 4)
        per_fold_probs.append(probs)
        y_pred_single = (probs > 0.5).sum(axis=1).astype(np.int64)
        metrics_single = compute_all_metrics(y_idrid, y_pred_single, num_classes=NUM_CLASSES)
        per_fold_metrics.append(metrics_single)
        print(f"    fold {fold_idx + 1} IDRiD test: QWK={metrics_single['qwk']:.4f}, "
              f"acc={metrics_single['accuracy']:.4f}, mae={metrics_single['mae']:.3f}")

        np.savez_compressed(
            out_dir / f"preds_fold{fold_idx + 1}.npz",
            idrid_head_probs=probs, y_true=y_idrid, y_pred=y_pred_single,
        )

    # --- Ensemble = mean P(y > k) across folds ---
    ensemble_probs = np.mean(np.stack(per_fold_probs, axis=0), axis=0)
    y_pred_ens = (ensemble_probs > 0.5).sum(axis=1).astype(np.int64)
    ens_metrics = compute_all_metrics(y_idrid, y_pred_ens, num_classes=NUM_CLASSES)
    print("\n" + "=" * 70)
    print("Cross-dataset:  APTOS 5-fold ensemble  →  IDRiD test")
    print(f"  QWK:      {ens_metrics['qwk']:.4f}")
    print(f"  Acc:      {ens_metrics['accuracy']:.4f}")
    print(f"  Macro F1: {ens_metrics['macro_f1']:.4f}")
    print(f"  MAE:      {ens_metrics['mae']:.3f}")

    # --- Summary ---
    per_fold_qwk = [m["qwk"] for m in per_fold_metrics]
    summary = {
        "protocol": "Train on APTOS 5-fold CV pool (ex-10% holdout); evaluate ensemble on IDRiD test.",
        "source": "aptos_train", "target": "idrid_test",
        "n_train": int(len(X_pool)), "n_test": int(len(X_idrid)),
        "seed": SEED, "epochs": EPOCHS, "patience": PATIENCE,
        "per_fold_test_qwk": per_fold_qwk,
        "per_fold_test_qwk_mean": float(np.mean(per_fold_qwk)),
        "per_fold_test_qwk_std": float(np.std(per_fold_qwk)),
        "ensemble_test": ens_metrics,
        "elapsed_sec": time.time() - t0,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    np.savez_compressed(out_dir / "ensemble.npz",
                        ensemble_head_probs=ensemble_probs,
                        y_true=y_idrid, y_pred=y_pred_ens)
    print(f"\nSaved -> {out_dir}/summary.json + ensemble.npz + preds_fold*.npz")


if __name__ == "__main__":
    main()
