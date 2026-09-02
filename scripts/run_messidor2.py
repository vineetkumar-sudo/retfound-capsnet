"""Day 9 — Within-Messidor-2 5-fold CV (full 5-class evaluation).

Structurally identical to scripts/run_idrid.py but:
  - Points at data/messidor2/features/.
  - No separate official test set; reports CV-val metrics only on 1,744
    gradable Messidor-2 images.
  - Reuses the Day 3 APTOS hyperparameters unchanged (the paper claim is about
    generalisation, not per-dataset tuning).

Per-model outputs at `results/messidor2/<model>/`:
    summary.json               pooled CV-val metrics
    preds_fold{N}.npz          y_true / y_pred + head arrays for UQ
    weights_fold{N}.pt         best-val-QWK state dict

Usage:
    uv run python scripts/run_messidor2.py                 # all 4 models
    uv run python scripts/run_messidor2.py --models ordinal_capsnet
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
from src.losses.ordinal_loss import OrdinalMarginLoss, predict_grade_from_heads
from src.models.baselines import MLPClassifier, MLPRegressor
from src.models.capsnet import CapsNet
from src.models.mlp_ordinal import MLPOrdinal
from src.models.ordinal_capsnet import OrdinalCapsNet
from src.utils import enable_tf32


FEATURES_DIR = Path("data/messidor2/features")
RESULTS_ROOT = Path("results/messidor2")
NUM_CLASSES = 5
FEATURE_DIM = 1024
N_FOLDS = 5
SEED = 42
EPOCHS = 100
PATIENCE = 30
BATCH_SIZE = 64
LR = 1e-3
WEIGHT_DECAY = 1e-4

MLP_HIDDEN = [512, 256]
MLP_DROPOUT = 0.3


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
        return CapsNet(feature_dim=FEATURE_DIM, num_classes=NUM_CLASSES,
                       num_primary=32, primary_dim=8, caps_dim=16, routing_iters=3)
    if model_id == "ordinal_capsnet":
        return OrdinalCapsNet(feature_dim=FEATURE_DIM, num_classes=NUM_CLASSES,
                              num_primary=32, primary_dim=8, caps_dim=16, routing_iters=3,
                              dropout=0.1)
    if model_id == "mlp_k1_sigmoid":
        # No-capsule comparator: same K-1 ordinal head-lengths schema so the
        # ordinal paths in predict()/train_one_fold() pick this up unchanged.
        return MLPOrdinal(feature_dim=FEATURE_DIM, hidden_dim=256,
                          num_classes=NUM_CLASSES, num_hidden_layers=2, dropout=0.1)
    raise ValueError(f"Unknown model: {model_id}")


@torch.no_grad()
def predict(model_id: str, model: nn.Module, loader: DataLoader,
            device: torch.device) -> dict:
    """Per-sample arrays; includes head arrays for UQ on ordinal/vanilla."""
    model.eval()
    ys, raw_chunks, extra_chunks = [], [], []
    extra_key = None
    rv_chunks: list[np.ndarray] = []
    for X, y in loader:
        X = X.to(device)
        if model_id in ("ordinal_capsnet", "mlp_k1_sigmoid"):
            out = model(X, return_routing_history=True)
            L = out["head_lengths"]
            rh = out["routing_history"]
            rv = rh.std(dim=1).mean(dim=(0, 2, 3)).cpu().numpy()
            extra_chunks.append(L.cpu().numpy())
            extra_key = "head_lengths"
            rv_chunks.append(rv)
            raw_chunks.append(predict_grade_from_heads(L).cpu().numpy().astype(np.int64))
        elif model_id == "capsnet_vanilla":
            out = model(X)
            lengths = out["lengths"]
            extra_chunks.append(lengths.cpu().numpy())
            extra_key = "class_lengths"
            raw_chunks.append(lengths.argmax(dim=1).cpu().numpy().astype(np.int64))
        elif model_id == "mlp_mse":
            pred = model(X).cpu().numpy()
            raw_chunks.append(np.clip(np.round(pred), 0, NUM_CLASSES - 1).astype(np.int64))
        else:  # mlp_ce
            logits = model(X).cpu().numpy()
            raw_chunks.append(logits.argmax(axis=1).astype(np.int64))
        ys.append(y.cpu().numpy())
    out = {
        "y_true": np.concatenate(ys).astype(np.int64),
        "y_pred": np.concatenate(raw_chunks),
    }
    if extra_chunks and extra_key:
        out[extra_key] = np.concatenate(extra_chunks, axis=0)
    if rv_chunks:
        out["routing_variance"] = np.concatenate(rv_chunks, axis=0).astype(np.float32)
    return out


def train_one_fold(model_id: str, model: nn.Module,
                   tr_loader: DataLoader, va_loader: DataLoader,
                   device: torch.device) -> tuple[dict, dict | None]:
    model = model.to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    if model_id == "mlp_ce":
        loss_fn = nn.CrossEntropyLoss()
    elif model_id == "mlp_mse":
        loss_fn = nn.MSELoss()
    elif model_id == "capsnet_vanilla":
        loss_fn = MarginLoss(num_classes=NUM_CLASSES)
    elif model_id in ("ordinal_capsnet", "mlp_k1_sigmoid"):
        loss_fn = OrdinalMarginLoss(num_classes=NUM_CLASSES)
    else:
        raise ValueError(model_id)

    best_qwk, best_state, best_metrics, no_improve = -1.0, None, {}, 0
    for epoch in range(1, EPOCHS + 1):
        model.train()
        for X, y in tr_loader:
            X = X.to(device); y = y.to(device)
            optim.zero_grad()
            if model_id == "mlp_mse":
                loss = loss_fn(model(X), y.float())
            elif model_id == "capsnet_vanilla":
                loss = loss_fn(model(X, labels=y)["lengths"], y)
            elif model_id in ("ordinal_capsnet", "mlp_k1_sigmoid"):
                loss = loss_fn(model(X)["head_lengths"], y)
            else:
                loss = loss_fn(model(X), y)
            loss.backward()
            optim.step()

        pv = predict(model_id, model, va_loader, device)
        m = compute_all_metrics(pv["y_true"], pv["y_pred"], num_classes=NUM_CLASSES)
        if m["qwk"] > best_qwk:
            best_qwk = m["qwk"]
            best_metrics = {**m, "best_epoch": epoch}
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
        if no_improve >= PATIENCE:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return best_metrics, best_state


def run_model(model_id: str, X, y, device: torch.device, out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    y_float = (model_id == "mlp_mse")
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

    per_fold_val: list[dict] = []
    t_global = time.time()
    for fold_idx, (tr_idx, va_idx) in enumerate(skf.split(X, y)):
        X_tr, X_va = X[tr_idx], X[va_idx]
        y_tr, y_va = y[tr_idx], y[va_idx]

        torch.manual_seed(SEED + fold_idx)
        np.random.seed(SEED + fold_idx)

        tr_loader, va_loader = make_loaders(X_tr, X_va, y_tr, y_va,
                                            BATCH_SIZE, y_float=y_float)
        model = build_model(model_id)
        t0 = time.time()
        val_m, state = train_one_fold(model_id, model, tr_loader, va_loader, device)
        dt = time.time() - t0

        pv = predict(model_id, model, va_loader, device)
        npz = {
            "y_true": pv["y_true"], "y_pred": pv["y_pred"],
        }
        for k in ("head_lengths", "class_lengths", "routing_variance"):
            if k in pv:
                npz[k] = pv[k]
        np.savez_compressed(out_dir / f"preds_fold{fold_idx + 1}.npz", **npz)
        if state is not None:
            torch.save(state, out_dir / f"weights_fold{fold_idx + 1}.pt")

        per_fold_val.append(val_m)
        print(f"  {model_id:<17s} fold {fold_idx + 1}/{N_FOLDS}  "
              f"val QWK={val_m['qwk']:.4f}  acc={val_m['accuracy']:.4f}  "
              f"(best@{val_m['best_epoch']}, {dt:.1f}s)")

    def _agg(key: str) -> tuple[float, float]:
        vs = [m[key] for m in per_fold_val]
        return float(np.mean(vs)), float(np.std(vs))

    result = {"model": model_id, "n_folds": N_FOLDS, "elapsed_sec": time.time() - t_global}
    for key in ("qwk", "accuracy", "macro_f1", "mae"):
        m, s = _agg(key)
        result[f"val_{key}_mean"] = m
        result[f"val_{key}_std"] = s
    pooled_cm = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    for m in per_fold_val:
        pooled_cm += np.asarray(m["confusion_matrix"])
    result["val_confusion_matrix_pooled"] = pooled_cm.tolist()
    result["per_fold_val_qwk"] = [m["qwk"] for m in per_fold_val]

    (out_dir / "summary.json").write_text(json.dumps(result, indent=2))
    print(f"  -> {model_id} DONE  val QWK {result['val_qwk_mean']:.4f} "
          f"± {result['val_qwk_std']:.4f}  ({result['elapsed_sec']:.1f}s total)")
    return result


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--models", nargs="+",
                   default=["mlp_ce", "mlp_mse", "capsnet_vanilla", "ordinal_capsnet"])
    p.add_argument("--features-dir", default=None,
                   help="Override feature cache directory (default data/messidor2/features). "
                        "Use data/messidor2/features_dinov2 for the DINOv2 backbone ablation.")
    p.add_argument("--output-dir", default=None,
                   help="Override output directory (default results/messidor2). "
                        "Use results/messidor2_dinov2 for the DINOv2 backbone ablation.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = get_device()
    features_dir = Path(args.features_dir) if args.features_dir else FEATURES_DIR
    out_root = Path(args.output_dir) if args.output_dir else RESULTS_ROOT
    print(f"Device: {device}")
    print(f"Features: {features_dir}")
    print(f"Output:   {out_root}")
    print(f"Models: {args.models}\n")

    X = np.load(features_dir / "features.npy")
    y = np.load(features_dir / "grades.npy")
    print(f"Messidor-2 (5-class): {X.shape}  label dist={np.bincount(y).tolist()}\n")

    out_root.mkdir(parents=True, exist_ok=True)
    summaries = {}
    for m in args.models:
        print(f"\n=== {m} ===")
        summaries[m] = run_model(m, X, y, device, out_root / m)

    # Console recap
    print("\n" + "=" * 80)
    print(f"{'Model':<22s}{'Val QWK':<22s}{'Val Acc':<22s}{'Val MAE':<10s}")
    for m, r in summaries.items():
        print(f"{m:<22s}"
              f"{r['val_qwk_mean']:.4f} ± {r['val_qwk_std']:.4f}    "
              f"{r['val_accuracy_mean']:.4f} ± {r['val_accuracy_std']:.4f}    "
              f"{r['val_mae_mean']:.3f}")


if __name__ == "__main__":
    main()
