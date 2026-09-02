"""Day 8 — IDRiD disease-grading runner.

Reuses the APTOS-tuned architectures and hyperparameters unchanged (the paper
claim is about generalization, not re-tuning). Trains 4 models on the 413-image
IDRiD training set with 5-fold stratified CV; for each fold's best checkpoint
(selected on val QWK), evaluates on the official 103-image test set.

Models (all sharing cached RETFound CLS features as input):
    mlp_ce          — MLPClassifier [512, 256] + dropout 0.3, cross-entropy
    mlp_mse         — MLPRegressor [512, 256] + dropout 0.3, round(clip) to grade
    capsnet_vanilla — 5-class CapsNet with Sabour margin loss
    ordinal_capsnet — K-1 binary-head Ordinal CapsNet (Day 3 champion)

Outputs per-model `results/idrid/<model>/`:
    summary.json                      pooled metrics (CV and test)
    preds_fold{N}.npz                 y_true, y_pred (+ head arrays for ordinal)
    weights_fold{N}.pt                state dict of best model (small, for reuse)

Usage:
    uv run python scripts/run_idrid.py                # all 4 models
    uv run python scripts/run_idrid.py --models ordinal_capsnet mlp_ce
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
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, TensorDataset

from src.evaluate import compute_all_metrics
from src.losses.margin_loss import MarginLoss
from src.losses.ordinal_loss import (
    OrdinalMarginLoss,
    predict_grade_from_heads,
)
from src.models.baselines import MLPClassifier, MLPRegressor
from src.models.capsnet import CapsNet
from src.models.ordinal_capsnet import OrdinalCapsNet
from src.utils import enable_tf32


FEATURES_DIR = Path("data/idrid/features")
RESULTS_ROOT = Path("results/idrid")
NUM_CLASSES = 5
FEATURE_DIM = 1024
N_FOLDS = 5
SEED = 42
EPOCHS = 100
PATIENCE = 30
BATCH_SIZE = 64
LR = 1e-3
WEIGHT_DECAY = 1e-4

# Baseline MLP shape matches configs/baselines.yaml for comparability
MLP_HIDDEN = [512, 256]
MLP_DROPOUT = 0.3


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

def get_device() -> torch.device:
    if torch.cuda.is_available():
        enable_tf32()
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def make_loaders(X_tr, X_va, y_tr, y_va, batch_size: int, y_float: bool = False):
    y_dtype = torch.float32 if y_float else torch.long
    tr = TensorDataset(torch.from_numpy(X_tr).float(), torch.from_numpy(y_tr).to(y_dtype))
    va = TensorDataset(torch.from_numpy(X_va).float(), torch.from_numpy(y_va).to(y_dtype))
    return (DataLoader(tr, batch_size=batch_size, shuffle=True),
            DataLoader(va, batch_size=batch_size, shuffle=False))


def build_model(model_id: str) -> nn.Module:
    if model_id == "mlp_ce":
        return MLPClassifier(FEATURE_DIM, MLP_HIDDEN, NUM_CLASSES, MLP_DROPOUT)
    if model_id == "mlp_mse":
        return MLPRegressor(FEATURE_DIM, MLP_HIDDEN, MLP_DROPOUT)
    if model_id == "capsnet_vanilla":
        return CapsNet(
            feature_dim=FEATURE_DIM, num_classes=NUM_CLASSES,
            num_primary=32, primary_dim=8, caps_dim=16, routing_iters=3,
        )
    if model_id == "ordinal_capsnet":
        return OrdinalCapsNet(
            feature_dim=FEATURE_DIM, num_classes=NUM_CLASSES,
            num_primary=32, primary_dim=8, caps_dim=16, routing_iters=3,
            dropout=0.1,
        )
    raise ValueError(f"Unknown model: {model_id}")


# ---------------------------------------------------------------------------
# Per-model eval / training
# ---------------------------------------------------------------------------

@torch.no_grad()
def predict(model_id: str, model: nn.Module, loader: DataLoader, device: torch.device) -> dict:
    """Return dict with y_true, y_pred, plus per-model extras (head_lengths for ordinal)."""
    model.eval()
    ys, raw_chunks, extra_chunks = [], [], []
    for X, y in loader:
        X = X.to(device)
        if model_id == "ordinal_capsnet":
            out = model(X)
            L = out["head_lengths"]
            extra_chunks.append(L.cpu().numpy())
            raw_chunks.append(predict_grade_from_heads(L).cpu().numpy().astype(np.int64))
        elif model_id == "capsnet_vanilla":
            out = model(X)
            lengths = out["lengths"]
            extra_chunks.append(lengths.cpu().numpy())
            raw_chunks.append(lengths.argmax(dim=1).cpu().numpy().astype(np.int64))
        elif model_id == "mlp_mse":
            pred = model(X).cpu().numpy()
            raw_chunks.append(np.clip(np.round(pred), 0, NUM_CLASSES - 1).astype(np.int64))
        else:  # mlp_ce
            logits = model(X).cpu().numpy()
            raw_chunks.append(logits.argmax(axis=1).astype(np.int64))
        ys.append(y.cpu().numpy())
    y_true = np.concatenate(ys).astype(np.int64)
    y_pred = np.concatenate(raw_chunks)
    out = {"y_true": y_true, "y_pred": y_pred}
    if extra_chunks:
        out["_extra"] = np.concatenate(extra_chunks, axis=0)
    return out


def train_one_fold(
    model_id: str, model: nn.Module, tr_loader: DataLoader, va_loader: DataLoader,
    device: torch.device, epochs: int, patience: int, lr: float, weight_decay: float,
) -> tuple[dict, dict | None]:
    """Train one fold, restore best-val-QWK state, return best_val_metrics + best_state dict."""
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Per-model loss function
    if model_id == "mlp_ce":
        loss_fn = nn.CrossEntropyLoss()
    elif model_id == "mlp_mse":
        loss_fn = nn.MSELoss()
    elif model_id == "capsnet_vanilla":
        loss_fn = MarginLoss(num_classes=NUM_CLASSES)
    elif model_id == "ordinal_capsnet":
        loss_fn = OrdinalMarginLoss(num_classes=NUM_CLASSES)
    else:
        raise ValueError(model_id)

    best_qwk, best_state, best_metrics, no_improve = -1.0, None, {}, 0

    for epoch in range(1, epochs + 1):
        model.train()
        for X, y in tr_loader:
            X = X.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            if model_id == "mlp_mse":
                pred = model(X)
                loss = loss_fn(pred, y.float())
            elif model_id == "capsnet_vanilla":
                out = model(X, labels=y)
                loss = loss_fn(out["lengths"], y)
            elif model_id == "ordinal_capsnet":
                out = model(X)
                loss = loss_fn(out["head_lengths"], y)
            else:
                loss = loss_fn(model(X), y)
            loss.backward()
            optimizer.step()

        # Val eval — use predict() + compute_all_metrics for consistency across models
        pv = predict(model_id, model, va_loader, device)
        m = compute_all_metrics(pv["y_true"], pv["y_pred"], num_classes=NUM_CLASSES)

        if m["qwk"] > best_qwk:
            best_qwk = m["qwk"]
            best_metrics = {**m, "best_epoch": epoch}
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
        if no_improve >= patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return best_metrics, best_state


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def run_model(
    model_id: str, X_train_all, y_train_all, X_test, y_test,
    device: torch.device, out_dir: Path,
) -> dict:
    """5-fold CV + official test eval for a single model. Writes preds, weights, summary.json."""
    out_dir.mkdir(parents=True, exist_ok=True)
    y_float = (model_id == "mlp_mse")

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

    per_fold_val, per_fold_test = [], []
    t_global = time.time()

    for fold_idx, (tr_idx, va_idx) in enumerate(skf.split(X_train_all, y_train_all)):
        X_tr, X_va = X_train_all[tr_idx], X_train_all[va_idx]
        y_tr, y_va = y_train_all[tr_idx], y_train_all[va_idx]

        torch.manual_seed(SEED + fold_idx)
        np.random.seed(SEED + fold_idx)

        tr_loader, va_loader = make_loaders(X_tr, X_va, y_tr, y_va, BATCH_SIZE, y_float=y_float)
        # Test loader (shared; batch doesn't matter for eval)
        _, test_loader = make_loaders(X_test[:1], X_test, y_test[:1], y_test,
                                      BATCH_SIZE, y_float=y_float)

        model = build_model(model_id)
        t0 = time.time()
        val_m, state = train_one_fold(
            model_id, model, tr_loader, va_loader, device,
            EPOCHS, PATIENCE, LR, WEIGHT_DECAY,
        )
        dt = time.time() - t0

        # Evaluate on official test
        pt = predict(model_id, model, test_loader, device)
        test_m = compute_all_metrics(pt["y_true"], pt["y_pred"], num_classes=NUM_CLASSES)

        # Save artifacts: preds (val + test) + weights
        npz = {
            "val_y_true": predict(model_id, model, va_loader, device)["y_true"],
            "val_y_pred": predict(model_id, model, va_loader, device)["y_pred"],
            "test_y_true": pt["y_true"],
            "test_y_pred": pt["y_pred"],
        }
        if "_extra" in pt:
            extra_name = "test_head_lengths" if model_id == "ordinal_capsnet" else "test_class_lengths"
            npz[extra_name] = pt["_extra"]
        np.savez_compressed(out_dir / f"preds_fold{fold_idx + 1}.npz", **npz)
        if state is not None:
            torch.save(state, out_dir / f"weights_fold{fold_idx + 1}.pt")

        per_fold_val.append(val_m)
        per_fold_test.append(test_m)
        print(f"  {model_id:<17s} fold {fold_idx + 1}/{N_FOLDS}  "
              f"val QWK={val_m['qwk']:.4f}  |  test QWK={test_m['qwk']:.4f} "
              f"(best@{val_m['best_epoch']}, {dt:.1f}s)")

    def _agg(metric_list: list[dict], key: str) -> tuple[float, float]:
        vs = [m[key] for m in metric_list]
        return float(np.mean(vs)), float(np.std(vs))

    result = {"model": model_id, "n_folds": N_FOLDS, "elapsed_sec": time.time() - t_global}
    for kind, ms in [("val", per_fold_val), ("test", per_fold_test)]:
        for key in ("qwk", "accuracy", "macro_f1", "mae"):
            m, s = _agg(ms, key)
            result[f"{kind}_{key}_mean"] = m
            result[f"{kind}_{key}_std"] = s
        # Pooled confusion matrix across folds (for per-class eval)
        pooled_cm = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
        for m in ms:
            pooled_cm += np.asarray(m["confusion_matrix"])
        result[f"{kind}_confusion_matrix_pooled"] = pooled_cm.tolist()
        result[f"per_fold_{kind}_qwk"] = [m["qwk"] for m in ms]
    (out_dir / "summary.json").write_text(json.dumps(result, indent=2))
    print(f"  -> {model_id} DONE: test QWK "
          f"{result['test_qwk_mean']:.4f} ± {result['test_qwk_std']:.4f} "
          f"({result['elapsed_sec']:.1f}s total)")
    return result


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--models", nargs="+",
                   default=["mlp_ce", "mlp_mse", "capsnet_vanilla", "ordinal_capsnet"])
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = get_device()
    print(f"Device: {device}")
    print(f"Models: {args.models}\n")

    X_train = np.load(FEATURES_DIR / "train_features.npy")
    y_train = np.load(FEATURES_DIR / "train_labels.npy")
    X_test = np.load(FEATURES_DIR / "test_features.npy")
    y_test = np.load(FEATURES_DIR / "test_labels.npy")
    print(f"Train: {X_train.shape} labels={np.bincount(y_train).tolist()}")
    print(f"Test:  {X_test.shape} labels={np.bincount(y_test).tolist()}\n")

    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    summaries = {}
    for m in args.models:
        print(f"\n=== {m} ===")
        summaries[m] = run_model(m, X_train, y_train, X_test, y_test, device,
                                 RESULTS_ROOT / m)

    # Compact side-by-side summary for quick inspection
    print("\n" + "=" * 78)
    print(f"{'Model':<20s}{'Val QWK':<22s}{'Test QWK':<22s}{'Test Acc':<18s}{'Test MAE':<10s}")
    for m, r in summaries.items():
        print(f"{m:<20s}"
              f"{r['val_qwk_mean']:.4f} ± {r['val_qwk_std']:.4f}    "
              f"{r['test_qwk_mean']:.4f} ± {r['test_qwk_std']:.4f}    "
              f"{r['test_accuracy_mean']:.4f} ± {r['test_accuracy_std']:.4f}  "
              f"{r['test_mae_mean']:.3f}")


if __name__ == "__main__":
    main()
