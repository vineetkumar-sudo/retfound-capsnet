"""Day 9 — Cross-dataset evaluation: train on APTOS, test on Messidor-2.

Runs all four Day-3-frozen model archetypes. For each archetype:
  1) Train 5 models on APTOS CV pool (ex 10% holdout), seed 42.
  2) For each fold, forward all 1,744 gradable Messidor-2 features through the
     best-val-QWK checkpoint.
  3) Ensemble across folds (mean of 5-class logits / mean of binary score).
  4) Report 5-class metrics (QWK / Acc / Macro F1 / MAE) AND binary metrics
     (Sensitivity, Specificity, AUC, F1) with referable = grade >= 2.

Binary continuous score per archetype (higher = more referable):
  mlp_ce         -> softmax(logits)[:, 2:].sum(axis=1)
  mlp_mse        -> raw regression output
  capsnet_vanilla-> lengths[:, 2] + lengths[:, 3] + lengths[:, 4]
  ordinal_capsnet-> primary = head_probs[:, 1] (P(y > 1));
                    also saved: head_probs.sum(axis=1)  (fallback severity)

Outputs:
  results/cross_dataset/aptos_to_messidor2/<model>/
      preds_fold{N}.npz
      ensemble.npz
      summary.json

Usage:
    uv run python scripts/cross_dataset_aptos_to_messidor2.py
    uv run python scripts/cross_dataset_aptos_to_messidor2.py --models ordinal_capsnet mlp_ce
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
from sklearn.metrics import f1_score, roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold, train_test_split
from torch.utils.data import DataLoader, TensorDataset

from src.evaluate import compute_all_metrics
from src.losses.margin_loss import MarginLoss
from src.losses.ordinal_loss import OrdinalMarginLoss, predict_grade_from_heads
from src.models.baselines import MLPClassifier, MLPRegressor
from src.models.capsnet import CapsNet
from src.models.ordinal_capsnet import OrdinalCapsNet


# Day 3 champion hyperparameters — unchanged across datasets.
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

MLP_HIDDEN = [512, 256]
MLP_DROPOUT = 0.3

MODEL_IDS = ("mlp_ce", "mlp_mse", "capsnet_vanilla", "ordinal_capsnet")


def get_device() -> torch.device:
    if torch.cuda.is_available(): return torch.device("cuda")
    if torch.backends.mps.is_available(): return torch.device("mps")
    return torch.device("cpu")


def make_loader(X, y, shuffle: bool, y_float: bool = False) -> DataLoader:
    y_dtype = torch.float32 if y_float else torch.long
    ds = TensorDataset(torch.from_numpy(X).float(), torch.from_numpy(y).to(y_dtype))
    return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=shuffle)


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
    raise ValueError(f"Unknown model: {model_id}")


@torch.no_grad()
def forward_messidor(model_id: str, model: nn.Module, loader: DataLoader,
                     device: torch.device) -> dict:
    """Run a single trained model on the Messidor-2 feature loader.

    Returns numpy arrays for:
      y_true, y_pred_5class, binary_score, and model-specific extras
      (head_lengths for ordinal, class_lengths for vanilla, logits/raw for MLPs,
      head_probs_sum as the fallback score for ordinal).
    """
    model.eval()
    ys, yp5, bscore, extras = [], [], [], {}
    for X, y in loader:
        X = X.to(device)
        if model_id == "ordinal_capsnet":
            out = model(X)
            L = out["head_lengths"]                  # (B, K-1, 2)
            head_probs = L[:, :, 1].cpu().numpy()    # (B, K-1)  P(y > k)
            yp5.append(predict_grade_from_heads(L).cpu().numpy().astype(np.int64))
            bscore.append(head_probs[:, 1])          # primary: P(y > 1)
            extras.setdefault("head_lengths", []).append(L.cpu().numpy())
            extras.setdefault("head_probs", []).append(head_probs)
            extras.setdefault("binary_score_sum_heads", []).append(head_probs.sum(axis=1))
        elif model_id == "capsnet_vanilla":
            out = model(X)
            lengths = out["lengths"].cpu().numpy()   # (B, K)
            yp5.append(lengths.argmax(axis=1).astype(np.int64))
            bscore.append(lengths[:, 2:].sum(axis=1))
            extras.setdefault("class_lengths", []).append(lengths)
        elif model_id == "mlp_mse":
            pred = model(X).cpu().numpy()            # raw regression scalar
            yp5.append(np.clip(np.round(pred), 0, NUM_CLASSES - 1).astype(np.int64))
            bscore.append(pred)
            extras.setdefault("raw", []).append(pred)
        else:  # mlp_ce
            logits = model(X).cpu().numpy()
            yp5.append(logits.argmax(axis=1).astype(np.int64))
            probs = np.exp(logits - logits.max(axis=1, keepdims=True))
            probs /= probs.sum(axis=1, keepdims=True)
            bscore.append(probs[:, 2:].sum(axis=1))
            extras.setdefault("logits", []).append(logits)
        ys.append(y.cpu().numpy())
    out = {
        "y_true": np.concatenate(ys).astype(np.int64),
        "y_pred_5class": np.concatenate(yp5),
        "binary_score": np.concatenate(bscore).astype(np.float32),
    }
    for k, v in extras.items():
        out[k] = np.concatenate(v, axis=0)
    return out


def train_one_fold(model_id: str, X_tr, y_tr, X_va, y_va,
                   device: torch.device, fold_idx: int) -> nn.Module:
    """Train one fold on APTOS; restore best-val-QWK state before returning."""
    torch.manual_seed(SEED + fold_idx)
    np.random.seed(SEED + fold_idx)

    y_float = (model_id == "mlp_mse")
    tr_loader = make_loader(X_tr, y_tr, shuffle=True, y_float=y_float)
    va_loader = make_loader(X_va, y_va, shuffle=False, y_float=y_float)

    model = build_model(model_id).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

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

    best_qwk, best_state, no_improve, best_epoch = -1.0, None, 0, 0
    for epoch in range(1, EPOCHS + 1):
        model.train()
        for X, y in tr_loader:
            X, y = X.to(device), y.to(device)
            optim.zero_grad()
            if model_id == "mlp_mse":
                loss = loss_fn(model(X), y.float())
            elif model_id == "capsnet_vanilla":
                loss = loss_fn(model(X, labels=y)["lengths"], y)
            elif model_id == "ordinal_capsnet":
                loss = loss_fn(model(X)["head_lengths"], y)
            else:
                loss = loss_fn(model(X), y)
            loss.backward()
            optim.step()
        # Val QWK
        model.eval()
        p = forward_messidor(model_id, model, va_loader, device)
        val_qwk = compute_all_metrics(p["y_true"], p["y_pred_5class"], num_classes=NUM_CLASSES)["qwk"]
        if val_qwk > best_qwk:
            best_qwk, best_epoch = val_qwk, epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
        if no_improve >= PATIENCE:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    print(f"    fold {fold_idx + 1}: best APTOS val QWK={best_qwk:.4f} @ ep{best_epoch}")
    return model


def binary_metrics(y_true_bin: np.ndarray, score: np.ndarray,
                   threshold: float = 0.5) -> dict:
    """Sens, Spec, AUC, F1 at a fixed decision threshold; also Youden-optimal threshold."""
    y_pred_bin = (score >= threshold).astype(int)
    tp = int(((y_pred_bin == 1) & (y_true_bin == 1)).sum())
    tn = int(((y_pred_bin == 0) & (y_true_bin == 0)).sum())
    fp = int(((y_pred_bin == 1) & (y_true_bin == 0)).sum())
    fn = int(((y_pred_bin == 0) & (y_true_bin == 1)).sum())
    sens = tp / (tp + fn) if (tp + fn) else float("nan")
    spec = tn / (tn + fp) if (tn + fp) else float("nan")
    try:
        auc = float(roc_auc_score(y_true_bin, score))
    except ValueError:
        auc = float("nan")
    f1 = float(f1_score(y_true_bin, y_pred_bin, zero_division=0))

    # Youden-optimal threshold (for reference; reported alongside the fixed-0.5 eval)
    try:
        fpr, tpr, thrs = roc_curve(y_true_bin, score)
        j = tpr - fpr
        best = int(np.argmax(j))
        y_thr = float(thrs[best])
        sens_y = float(tpr[best]); spec_y = float(1 - fpr[best])
    except ValueError:
        y_thr, sens_y, spec_y = float("nan"), float("nan"), float("nan")

    return {
        "threshold": float(threshold),
        "sensitivity": float(sens), "specificity": float(spec),
        "auc": auc, "f1": f1,
        "confusion": {"tp": tp, "tn": tn, "fp": fp, "fn": fn},
        "youden_threshold": y_thr, "youden_sensitivity": sens_y, "youden_specificity": spec_y,
    }


def run_model(model_id: str, X_pool, y_pool, X_mess, y_mess_5, y_mess_bin,
              device: torch.device, out_root: Path) -> dict:
    out_dir = out_root / model_id
    out_dir.mkdir(parents=True, exist_ok=True)
    y_float_mess = (model_id == "mlp_mse")
    # Labels on the Messidor side are 5-class ints; for mlp_mse we still store as long to match forward loop
    mess_labels = y_mess_5 if not y_float_mess else y_mess_5  # dtype conversion happens in make_loader

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    t0 = time.time()
    per_fold_binary_scores: list[np.ndarray] = []
    per_fold_y_pred_5class: list[np.ndarray] = []
    per_fold_metrics: list[dict] = []
    y_true_all: np.ndarray | None = None

    for fold_idx, (tr_idx, va_idx) in enumerate(skf.split(X_pool, y_pool)):
        print(f"\n=== {model_id}  fold {fold_idx + 1}/{N_FOLDS} ===")
        model = train_one_fold(model_id, X_pool[tr_idx], y_pool[tr_idx],
                               X_pool[va_idx], y_pool[va_idx],
                               device, fold_idx)
        mess_loader = make_loader(X_mess, mess_labels, shuffle=False,
                                  y_float=y_float_mess)
        preds = forward_messidor(model_id, model, mess_loader, device)

        # Use the true Messidor-2 5-class labels for scoring (loader may have forced float labels for mse)
        y_true_all = y_mess_5
        m5 = compute_all_metrics(y_mess_5, preds["y_pred_5class"], num_classes=NUM_CLASSES)
        b = binary_metrics(y_mess_bin, preds["binary_score"])
        per_fold_metrics.append({"5class": m5, "binary": b})
        per_fold_binary_scores.append(preds["binary_score"])
        per_fold_y_pred_5class.append(preds["y_pred_5class"])

        # Save per-fold preds
        payload = {
            "y_true_5class": y_mess_5, "y_true_binary": y_mess_bin,
            "y_pred_5class": preds["y_pred_5class"],
            "binary_score": preds["binary_score"],
        }
        for k in ("head_lengths", "head_probs", "binary_score_sum_heads",
                  "class_lengths", "logits", "raw"):
            if k in preds:
                payload[k] = preds[k]
        np.savez_compressed(out_dir / f"preds_fold{fold_idx + 1}.npz", **payload)
        print(f"    fold {fold_idx + 1} Messidor-2: 5-class QWK={m5['qwk']:.4f}  |  "
              f"binary AUC={b['auc']:.4f}  sens={b['sensitivity']:.3f}  spec={b['specificity']:.3f}")

    # --- Ensemble ---
    ensemble_binary_score = np.mean(np.stack(per_fold_binary_scores, axis=0), axis=0)
    # 5-class ensemble: majority vote across folds (simple + fold-count-agnostic).
    ypred_stack = np.stack(per_fold_y_pred_5class, axis=0)  # (folds, N)
    ensemble_y_pred_5class = np.zeros(ypred_stack.shape[1], dtype=np.int64)
    for i in range(ypred_stack.shape[1]):
        vals, counts = np.unique(ypred_stack[:, i], return_counts=True)
        ensemble_y_pred_5class[i] = int(vals[np.argmax(counts)])

    assert y_true_all is not None
    ens_5 = compute_all_metrics(y_true_all, ensemble_y_pred_5class, num_classes=NUM_CLASSES)
    ens_b = binary_metrics(y_mess_bin, ensemble_binary_score)

    summary = {
        "model": model_id,
        "protocol": "Train APTOS 5-fold CV pool (ex 10% holdout); eval ensemble on Messidor-2 (1744 gradable).",
        "n_folds": N_FOLDS, "n_test": int(len(y_true_all)),
        "per_fold": per_fold_metrics,
        "per_fold_5class_qwk": [m["5class"]["qwk"] for m in per_fold_metrics],
        "per_fold_binary_auc": [m["binary"]["auc"] for m in per_fold_metrics],
        "ensemble_5class": ens_5,
        "ensemble_binary": ens_b,
        "elapsed_sec": time.time() - t0,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    np.savez_compressed(
        out_dir / "ensemble.npz",
        y_true_5class=y_true_all, y_true_binary=y_mess_bin,
        y_pred_5class=ensemble_y_pred_5class,
        binary_score=ensemble_binary_score,
    )
    print(f"\n  -> {model_id} DONE  ensemble 5-class QWK={ens_5['qwk']:.4f}  "
          f"binary AUC={ens_b['auc']:.4f}  ({summary['elapsed_sec']:.1f}s)")
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=list(MODEL_IDS))
    args = ap.parse_args()

    device = get_device()
    print(f"Device: {device}")
    print(f"Models: {args.models}\n")

    # --- APTOS pool (ex-10% holdout, seed 42) ---
    X_aptos = np.load("data/aptos/features/train_features.npy")
    y_aptos = np.load("data/aptos/features/train_labels.npy")
    pool_idx, _ = train_test_split(
        np.arange(len(X_aptos)), test_size=HOLDOUT_FRAC,
        stratify=y_aptos, random_state=SEED,
    )
    X_pool, y_pool = X_aptos[pool_idx], y_aptos[pool_idx]
    print(f"APTOS CV pool: {X_pool.shape[0]}")

    # --- Messidor-2 features (only gradable rows, prepared by extract_messidor2_features.py) ---
    X_mess = np.load("data/messidor2/features/features.npy")
    y_mess_5 = np.load("data/messidor2/features/grades.npy")
    y_mess_bin = np.load("data/messidor2/features/binary_labels.npy")
    print(f"Messidor-2:   {X_mess.shape[0]} gradable  grade-dist={np.bincount(y_mess_5, minlength=5).tolist()}  "
          f"binary-dist={np.bincount(y_mess_bin).tolist()}\n")

    out_root = Path("results/cross_dataset/aptos_to_messidor2")
    out_root.mkdir(parents=True, exist_ok=True)

    all_summaries: dict[str, dict] = {}
    for m in args.models:
        if m not in MODEL_IDS:
            raise ValueError(f"Unknown model id: {m}")
        all_summaries[m] = run_model(m, X_pool, y_pool, X_mess, y_mess_5, y_mess_bin,
                                     device, out_root)

    # Console summary
    print("\n" + "=" * 86)
    print(f"{'Model':<20s}{'5-class QWK':<16s}{'5-class Acc':<16s}{'Binary AUC':<14s}"
          f"{'Sens':<10s}{'Spec':<10s}{'F1':<8s}")
    for m, s in all_summaries.items():
        e5, eb = s["ensemble_5class"], s["ensemble_binary"]
        print(f"{m:<20s}{e5['qwk']:<16.4f}{e5['accuracy']:<16.4f}{eb['auc']:<14.4f}"
              f"{eb['sensitivity']:<10.4f}{eb['specificity']:<10.4f}{eb['f1']:<8.4f}")


if __name__ == "__main__":
    main()
