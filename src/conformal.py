"""Split conformal prediction for K-1 ordinal DR grading heads.

Two nonconformity scores are implemented: LAC (Least Ambiguous Classifier,
Sadinle et al. 2019) and APS (Adaptive Prediction Sets, Romano et al. 2020).
Both operate on the chain-rule 5-class probability matrix derived from the
K-1 head outputs; nothing here is specific to capsules vs. sigmoid heads.

Conformal prediction gives a distribution-free marginal coverage guarantee:
for any miscoverage level alpha, the prediction-set C(X) contains the true
label with probability at least 1 - alpha, provided the calibration and
test samples are exchangeable. This is the UQ story that *survives* the
poor ECE we observed in src/calibration.py — conformal does not require
the probabilities to be calibrated, only that they are exchangeable between
cal and test.

Core helpers:

  * lac_scores(probs, y) -> 1 - probs[i, y[i]]
    "How unsure is the model that the true class is y[i]?"

  * aps_scores(probs, y) -> cumulative probability up to y[i] in descending
                            order of class probability, with a tie-breaking
                            uniform jitter to avoid atoms at the boundary.

  * build_prediction_sets(probs, q_hat, score="lac") -> (N, K) bool mask

  * evaluate_conformal(probs, y, alpha=0.1, score="lac", split_seed=42)
      -> dict with marginal coverage, class-conditional coverage, mean
      set size, set-size histogram. Uses a single random cal/test split
      (50/50 by default) with `split_seed` for reproducibility.

All routines operate on numpy; no torch dependency.
"""

from __future__ import annotations

import numpy as np


def lac_scores(probs: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Nonconformity of each (x, y) pair under the LAC score.

    Args:
        probs: (N, K) chain-rule probabilities, rows sum to 1.
        y: (N,) int true labels in [0, K-1].
    Returns:
        (N,) nonconformity in [0, 1]; larger means less conforming.
    """
    n = len(y)
    return 1.0 - probs[np.arange(n), y]


def aps_scores(
    probs: np.ndarray, y: np.ndarray, rng: np.random.Generator | None = None
) -> np.ndarray:
    """APS nonconformity score with a randomised uniform tie-breaker.

    For each sample we sort classes in descending probability order, sum
    probabilities from the top until we hit the true class y[i], then
    subtract a uniform fraction u[i] * p_sorted[rank of y[i]] (Romano 2020
    trick) to randomise the cumulative boundary. This prevents systematic
    over-coverage on ties.
    """
    rng = rng or np.random.default_rng(0)
    n, k = probs.shape
    order = np.argsort(-probs, axis=1, kind="stable")                 # (N, K)
    sorted_probs = np.take_along_axis(probs, order, axis=1)           # (N, K)
    cum = np.cumsum(sorted_probs, axis=1)                             # (N, K)

    # Rank of each true class in the sorted order.
    rank_of_y = np.zeros(n, dtype=np.int64)
    for i in range(n):
        rank_of_y[i] = int(np.where(order[i] == y[i])[0][0])

    cum_at_y = cum[np.arange(n), rank_of_y]
    p_at_y = sorted_probs[np.arange(n), rank_of_y]
    u = rng.random(n)
    return cum_at_y - u * p_at_y


def _quantile_hi(scores: np.ndarray, alpha: float) -> float:
    """Finite-sample-corrected upper quantile used by split conformal."""
    n = len(scores)
    k = int(np.ceil((n + 1) * (1 - alpha)))
    k = min(max(k, 1), n)
    return float(np.sort(scores)[k - 1])


def build_prediction_sets_lac(probs: np.ndarray, q_hat: float) -> np.ndarray:
    """Include class k if probs[i, k] >= 1 - q_hat, for every i."""
    return probs >= (1.0 - q_hat)


def build_prediction_sets_aps(
    probs: np.ndarray, q_hat: float, rng: np.random.Generator | None = None
) -> np.ndarray:
    """APS prediction set: include classes up to cumulative prob >= q_hat.

    We include every class k whose running cumulative probability (in
    descending order) does not exceed q_hat, plus the next class, minus a
    uniform random fraction of that next class's probability. This matches
    the Romano 2020 construction and preserves exchangeability with the
    cal-set APS scores produced by `aps_scores`.
    """
    rng = rng or np.random.default_rng(0)
    n, k = probs.shape
    order = np.argsort(-probs, axis=1, kind="stable")
    sorted_probs = np.take_along_axis(probs, order, axis=1)
    cum = np.cumsum(sorted_probs, axis=1)

    # For each row, find the smallest rank whose cum >= q_hat.
    # Randomised inclusion/exclusion of that boundary class.
    sets = np.zeros((n, k), dtype=bool)
    u = rng.random(n)
    for i in range(n):
        rank = int(np.searchsorted(cum[i], q_hat, side="left"))
        if rank == k:                       # all classes needed
            sets[i, :] = True
            continue
        keep_ranks = list(range(rank))      # strictly below q_hat
        # Randomised inclusion of the boundary class: include it iff the
        # uniform draw would keep the cumulative below q_hat even after
        # subtracting u_i * p_rank (mirrors aps_scores' randomisation).
        include_boundary = (cum[i, rank] - u[i] * sorted_probs[i, rank]) <= q_hat
        if include_boundary:
            keep_ranks.append(rank)
        if not keep_ranks:                  # guarantee nonempty set
            keep_ranks = [0]
        for r in keep_ranks:
            sets[i, order[i, r]] = True
    return sets


def evaluate_conformal(
    probs: np.ndarray,
    y: np.ndarray,
    alpha: float = 0.10,
    score: str = "lac",
    split_seed: int = 42,
    cal_frac: float = 0.5,
) -> dict:
    """Single-split conformal evaluation.

    Args:
        probs: (N, K) chain-rule probability matrix, rows sum to 1.
        y: (N,) int true labels.
        alpha: miscoverage level (e.g. 0.10 for 90% coverage).
        score: "lac" or "aps".
        split_seed: RNG seed for cal/test split + APS jitter.
        cal_frac: fraction of data used for calibration (rest for test).
    Returns:
        dict with marginal_coverage, mean_set_size, class_conditional_coverage
        (K-vector), set_size_histogram (K+1-vector of counts), q_hat, alpha,
        n_cal, n_test.
    """
    assert score in ("lac", "aps")
    rng = np.random.default_rng(split_seed)
    n = len(y)
    idx = rng.permutation(n)
    n_cal = int(round(cal_frac * n))
    cal_idx, test_idx = idx[:n_cal], idx[n_cal:]

    probs_cal, y_cal = probs[cal_idx], y[cal_idx]
    probs_test, y_test = probs[test_idx], y[test_idx]

    if score == "lac":
        cal_scores = lac_scores(probs_cal, y_cal)
        q_hat = _quantile_hi(cal_scores, alpha)
        sets = build_prediction_sets_lac(probs_test, q_hat)
    else:
        cal_scores = aps_scores(probs_cal, y_cal, rng=rng)
        q_hat = _quantile_hi(cal_scores, alpha)
        sets = build_prediction_sets_aps(probs_test, q_hat, rng=rng)

    covered = sets[np.arange(len(y_test)), y_test]                   # (N_test,)
    marginal = float(covered.mean())
    set_sizes = sets.sum(axis=1)
    mean_set_size = float(set_sizes.mean())
    k = probs.shape[1]
    size_hist = [int((set_sizes == s).sum()) for s in range(k + 1)]

    ccov: list[float] = []
    for c in range(k):
        mask = (y_test == c)
        if mask.sum() == 0:
            ccov.append(float("nan"))
        else:
            ccov.append(float(covered[mask].mean()))

    return {
        "alpha": alpha,
        "score": score,
        "q_hat": q_hat,
        "n_cal": int(n_cal),
        "n_test": int(n - n_cal),
        "marginal_coverage": marginal,
        "mean_set_size": mean_set_size,
        "class_conditional_coverage": ccov,
        "set_size_histogram": size_hist,
        "split_seed": split_seed,
    }
