"""Calibration + selective-prediction metrics for K-1 ordinal heads.

All helpers operate on the standard `head_lengths` array (N, K-1, 2) saved by
both the APTOS runner (run_ordinal_capsnet.py) and the Messidor-2 runner
(run_messidor2.py). The downstream chain-rule 5-class probabilities are
derived once here so every metric (ECE, MCE, AURC, risk@coverage) uses the
same probability-of-class estimator.

Metrics implemented (each with a one-line spec):

  * chain_rule_probs(head_probs) -> (N, K) valid-probability matrix
    derived from the K-1 "P(y > k)" heads, with optional renormalisation.

  * expected_calibration_error(confs, correct, n_bins=10, mass=True)
    Weighted |acc - conf| across confidence bins. mass=True uses equal-mass
    bins (recommended by Nixon et al. 2019); False uses equal-width.

  * maximum_calibration_error(confs, correct, n_bins=10, mass=True)
    The worst bin's |acc - conf|.

  * reliability_bins(confs, correct, n_bins=10, mass=True)
    Per-bin (mean_conf, mean_acc, weight) for reliability diagrams.

  * aurc(confs, correct) -> (aurc, excess_aurc)
    Area under risk-coverage curve (lower is better). Risk = 1 - accuracy
    on the samples retained after rejecting the least-confident. Excess-AURC
    is AURC minus the oracle AURC (same number of errors, ranked last).

  * risk_at_coverage(confs, correct, coverage) -> float
    Error rate when we retain the top-`coverage` fraction by confidence.

Conventions:
  * `confs` is a 1-D array of per-sample confidence in [0, 1] -- higher means
    "more certain". For multi-class we use max chain-rule probability.
  * `correct` is a 0/1 int array of per-sample correctness.
  * All returns are plain Python floats so JSON serialisation is one-step.
"""

from __future__ import annotations

import numpy as np


def chain_rule_probs(
    head_probs: np.ndarray, num_classes: int = 5, renormalise: bool = True
) -> tuple[np.ndarray, float]:
    """Convert K-1 "P(y>k)" head probabilities to a K-class probability matrix.

    For K=5: P(y=0) = 1 - P(y>0); P(y=k) = P(y>k-1) * (1 - P(y>k)) for k>=1;
    P(y=K-1) = P(y>K-2). When the heads are non-monotonic (any P(y>k+1) >
    P(y>k)) the raw chain-rule product can yield negative values; we clip at
    zero and optionally renormalise so the row sums to 1. The second return
    value is the fraction of samples that required any non-trivial
    renormalisation (an interpretable metric of ordinal consistency).

    Args:
        head_probs: (N, K-1) float array in [0, 1], P(y>k).
        num_classes: K (default 5).
        renormalise: if True, clip to 0 and divide by per-row sum so each
                     row is a proper probability distribution. Keep False if
                     you want to characterise monotonicity violations.
    Returns:
        probs: (N, K) float64 probability matrix.
        renorm_rate: fraction of rows where pre-normalisation sum was
                     != 1 by more than 1e-4 (i.e. the chain-rule product did
                     not give a valid distribution natively).
    """
    h = np.asarray(head_probs, dtype=np.float64)        # (N, K-1)
    n, k_minus_1 = h.shape
    assert k_minus_1 == num_classes - 1, \
        f"head_probs last dim {k_minus_1} != num_classes-1 = {num_classes - 1}"

    probs = np.zeros((n, num_classes), dtype=np.float64)
    probs[:, 0] = 1.0 - h[:, 0]
    for k in range(1, num_classes - 1):
        probs[:, k] = h[:, k - 1] * (1.0 - h[:, k])
    probs[:, -1] = h[:, -1]

    probs = np.clip(probs, 0.0, None)
    row_sums = probs.sum(axis=1)
    renorm_needed = np.abs(row_sums - 1.0) > 1e-4
    renorm_rate = float(renorm_needed.mean())
    if renormalise:
        probs = probs / np.maximum(row_sums, 1e-12)[:, None]
    return probs, renorm_rate


def max_confidence(probs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-sample (max_prob, argmax_class)."""
    return probs.max(axis=1), probs.argmax(axis=1)


def _bin_edges(confs: np.ndarray, n_bins: int, mass: bool) -> np.ndarray:
    if mass:
        qs = np.linspace(0.0, 1.0, n_bins + 1)
        edges = np.quantile(confs, qs)
        edges[0] = min(edges[0], 0.0)
        edges[-1] = max(edges[-1], 1.0)
        return edges
    return np.linspace(0.0, 1.0, n_bins + 1)


def reliability_bins(
    confs: np.ndarray, correct: np.ndarray, n_bins: int = 10, mass: bool = True
) -> list[dict]:
    """Per-bin (count, mean_conf, mean_acc, weight) for reliability diagrams."""
    edges = _bin_edges(confs, n_bins, mass)
    out: list[dict] = []
    n = len(confs)
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        in_bin = (confs >= lo) & (confs <= hi) if i == n_bins - 1 else \
                 (confs >= lo) & (confs < hi)
        count = int(in_bin.sum())
        if count == 0:
            out.append({"lo": float(lo), "hi": float(hi), "count": 0,
                        "mean_conf": float("nan"), "mean_acc": float("nan"),
                        "weight": 0.0})
            continue
        mean_conf = float(confs[in_bin].mean())
        mean_acc = float(correct[in_bin].mean())
        out.append({
            "lo": float(lo), "hi": float(hi), "count": count,
            "mean_conf": mean_conf, "mean_acc": mean_acc,
            "weight": count / n,
        })
    return out


def expected_calibration_error(
    confs: np.ndarray, correct: np.ndarray, n_bins: int = 10, mass: bool = True
) -> float:
    bins = reliability_bins(confs, correct, n_bins=n_bins, mass=mass)
    return float(sum(b["weight"] * abs(b["mean_acc"] - b["mean_conf"])
                     for b in bins if b["count"] > 0))


def maximum_calibration_error(
    confs: np.ndarray, correct: np.ndarray, n_bins: int = 10, mass: bool = True
) -> float:
    bins = reliability_bins(confs, correct, n_bins=n_bins, mass=mass)
    gaps = [abs(b["mean_acc"] - b["mean_conf"]) for b in bins if b["count"] > 0]
    return float(max(gaps)) if gaps else 0.0


def aurc(confs: np.ndarray, correct: np.ndarray) -> tuple[float, float]:
    """Area under the risk-coverage curve, and the excess over an oracle.

    Risk(c) = error rate on the top-`c` fraction of samples ranked by
    confidence (highest first). AURC = mean_c risk(c) over all discrete
    coverage levels c = 1/N, 2/N, ..., 1. Excess-AURC = AURC - oracle_AURC
    where the oracle puts all errors at the bottom of the rank; this makes
    the number comparable across datasets with different base error rates.

    Returns: (aurc, excess_aurc), both in [0, 1].
    """
    n = len(confs)
    order = np.argsort(-confs, kind="stable")          # highest confidence first
    c_sorted = correct[order]
    cum_err = np.cumsum(1 - c_sorted)
    coverage_risks = cum_err / np.arange(1, n + 1)     # (N,)
    aurc_val = float(coverage_risks.mean())

    # Oracle: same total errors, but placed at the end of the ranking.
    n_err = int((1 - correct).sum())
    n_correct = n - n_err
    oracle_risks = np.concatenate([
        np.zeros(n_correct),
        np.arange(1, n_err + 1) / (n_correct + np.arange(1, n_err + 1)),
    ]) if n_err > 0 else np.zeros(n)
    oracle_aurc = float(oracle_risks.mean())

    return aurc_val, aurc_val - oracle_aurc


def risk_at_coverage(
    confs: np.ndarray, correct: np.ndarray, coverage: float
) -> float:
    """Error rate when retaining the top-`coverage` fraction by confidence."""
    assert 0.0 < coverage <= 1.0
    n = len(confs)
    k = max(1, int(round(coverage * n)))
    order = np.argsort(-confs, kind="stable")
    retained_correct = correct[order[:k]]
    return float(1.0 - retained_correct.mean())


def predict_grade_from_head_lengths(head_lengths: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """Numpy twin of src.losses.ordinal_loss.predict_grade_from_heads."""
    p_gt = head_lengths[:, :, 1]
    return (p_gt > threshold).sum(axis=1).astype(np.int64)
