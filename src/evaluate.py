"""Shared metric helpers for APTOS ablation table + figures.

All scripts should call `compute_all_metrics` instead of inlining sklearn calls,
so QWK / accuracy / macro-F1 / MAE are consistent everywhere.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
)


def compute_all_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    num_classes: int = 5,
) -> dict:
    """Full APTOS metric bundle. y_true / y_pred are 1-D int arrays in [0, num_classes)."""
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    labels = list(range(num_classes))
    return {
        "qwk": float(cohen_kappa_score(y_true, y_pred, weights="quadratic", labels=labels)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", labels=labels, zero_division=0)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
    }


def mae_from_confusion_matrix(cm: np.ndarray) -> float:
    """Recover MAE from a confusion matrix when raw preds aren't available.
    MAE = sum_{i,j} |i - j| * CM[i,j] / sum(CM)."""
    cm = np.asarray(cm, dtype=float)
    n = int(cm.shape[0])
    idx = np.arange(n)
    weights = np.abs(idx[:, None] - idx[None, :])
    total = cm.sum()
    if total == 0:
        return float("nan")
    return float((cm * weights).sum() / total)
