"""Extended calibration metrics: temperature scaling + Brier + NLL.

The original `src.calibration` module provides ECE / MCE / AURC / reliability
bins. This module adds the post-hoc calibration baselines the 2024--2026
calibration literature expects alongside ECE: (1) temperature scaling (TS,
a single-parameter post-hoc fix on the cal split), (2) Brier score, and
(3) negative log-likelihood. All three operate on the same chain-rule
probability matrix and are computed without retraining.

Key design:

  * `fit_temperature(probs_cal, y_cal)` -> scalar T.
      Gradient-free 1-D optimiser (golden-section on log T) minimising NLL
      on the calibration split. Works on per-sample chain-rule probabilities
      by converting to logits first (probs are clipped to (eps, 1-eps)).

  * `apply_temperature(probs, T)` -> temperature-scaled probabilities.

  * `brier_score(probs, y, num_classes)` -> multi-class Brier (one-hot y).

  * `nll(probs, y)` -> cross-entropy / NLL, per-sample mean.

  * `temperature_scale_eval(probs_cal, y_cal, probs_test, y_test, y_pred_test)`
    -> dict with pre-TS / post-TS ECE, Brier, NLL, plus the fitted T.

All computations are pure numpy; no torch, no gradients.
"""

from __future__ import annotations

import numpy as np

from src.calibration import (
    expected_calibration_error,
    maximum_calibration_error,
    reliability_bins,
)


_EPS = 1e-7


def _probs_to_logits(probs: np.ndarray) -> np.ndarray:
    """Invert a softmax-like normalisation: treat log p as logits (up to const).

    This is not the ``true'' underlying logits of the K-1 capsule or sigmoid
    head -- we don't have those at this point -- but it is the right surrogate
    for temperature scaling because shifting by a constant is absorbed by the
    softmax renormalisation, and only the relative spacing of log probs matters.
    """
    p = np.clip(probs, _EPS, 1.0 - _EPS)
    return np.log(p)


def _softmax_with_T(logits: np.ndarray, T: float) -> np.ndarray:
    z = logits / max(T, _EPS)
    z = z - z.max(axis=1, keepdims=True)
    ez = np.exp(z)
    return ez / ez.sum(axis=1, keepdims=True)


def nll(probs: np.ndarray, y: np.ndarray) -> float:
    """Per-sample mean negative log-likelihood."""
    p = np.clip(probs[np.arange(len(y)), y], _EPS, 1.0)
    return float(-np.log(p).mean())


def brier_score(probs: np.ndarray, y: np.ndarray, num_classes: int = 5) -> float:
    """Multi-class Brier score: mean squared distance from one-hot target."""
    onehot = np.zeros_like(probs)
    onehot[np.arange(len(y)), y] = 1.0
    return float(np.mean(np.sum((probs - onehot) ** 2, axis=1)))


def fit_temperature(
    probs_cal: np.ndarray, y_cal: np.ndarray,
    log_T_min: float = -2.0, log_T_max: float = 2.0, n_steps: int = 50,
) -> float:
    """Find T that minimises NLL on the calibration split.

    Uses a grid search in log space (cheap, ~50 evals) followed by a
    golden-section refine. No gradients; robust to any probability input.
    """
    logits = _probs_to_logits(probs_cal)
    log_Ts = np.linspace(log_T_min, log_T_max, n_steps)
    nlls = []
    for lT in log_Ts:
        T = float(np.exp(lT))
        p = _softmax_with_T(logits, T)
        nlls.append(nll(p, y_cal))
    best_i = int(np.argmin(nlls))

    # Golden-section refine in a narrow window around the grid minimum.
    lo = log_Ts[max(0, best_i - 1)]
    hi = log_Ts[min(n_steps - 1, best_i + 1)]
    phi = (1 + 5 ** 0.5) / 2
    for _ in range(40):
        a = hi - (hi - lo) / phi
        b = lo + (hi - lo) / phi
        T_a, T_b = float(np.exp(a)), float(np.exp(b))
        n_a = nll(_softmax_with_T(logits, T_a), y_cal)
        n_b = nll(_softmax_with_T(logits, T_b), y_cal)
        if n_a < n_b:
            hi = b
        else:
            lo = a
        if abs(hi - lo) < 1e-4:
            break
    return float(np.exp((lo + hi) / 2))


def apply_temperature(probs: np.ndarray, T: float) -> np.ndarray:
    return _softmax_with_T(_probs_to_logits(probs), T)


def temperature_scale_eval(
    probs: np.ndarray,
    y: np.ndarray,
    y_pred: np.ndarray,
    num_classes: int = 5,
    cal_frac: float = 0.5,
    split_seed: int = 42,
) -> dict:
    """End-to-end TS evaluation on a single cal/test split.

    Fits T on a random 50% calibration fold, measures ECE / Brier / NLL on
    the remaining 50% both before and after scaling. Confidence for ECE is
    P(y = predicted_grade), matching the convention in src.calibration.
    """
    rng = np.random.default_rng(split_seed)
    idx = rng.permutation(len(y))
    n_cal = int(round(cal_frac * len(y)))
    cal, test = idx[:n_cal], idx[n_cal:]

    probs_cal, probs_test = probs[cal], probs[test]
    y_cal, y_test = y[cal], y[test]
    y_pred_test = y_pred[test]
    correct = (y_pred_test == y_test).astype(np.int64)

    T = fit_temperature(probs_cal, y_cal)
    probs_test_ts = apply_temperature(probs_test, T)

    confs_pre = probs_test[np.arange(len(y_test)), y_pred_test]
    confs_post = probs_test_ts[np.arange(len(y_test)), y_pred_test]

    return {
        "T": T,
        "ece_pre": expected_calibration_error(confs_pre, correct),
        "ece_post": expected_calibration_error(confs_post, correct),
        "mce_pre": maximum_calibration_error(confs_pre, correct),
        "mce_post": maximum_calibration_error(confs_post, correct),
        "brier_pre": brier_score(probs_test, y_test, num_classes),
        "brier_post": brier_score(probs_test_ts, y_test, num_classes),
        "nll_pre": nll(probs_test, y_test),
        "nll_post": nll(probs_test_ts, y_test),
        "n_cal": int(n_cal),
        "n_test": int(len(y_test)),
    }
