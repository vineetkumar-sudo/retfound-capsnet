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


def raps_scores(
    probs: np.ndarray,
    y: np.ndarray,
    k_reg: int = 1,
    lambda_reg: float = 0.01,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Regularised APS score (Angelopoulos, Bates, Malik, Jordan, ICLR 2021).

    Like APS but adds a penalty `lambda_reg * max(0, rank - k_reg)` to each
    class's contribution, shrinking prediction sets. k_reg=1 with a small
    lambda is the common default; setting lambda_reg=0 recovers plain APS.
    """
    rng = rng or np.random.default_rng(0)
    n, k = probs.shape
    order = np.argsort(-probs, axis=1, kind="stable")
    sorted_probs = np.take_along_axis(probs, order, axis=1)
    # Regularised cumulative: each class adds its prob + a penalty if its rank
    # exceeds k_reg.
    ranks = np.arange(k, dtype=np.float64)
    reg = lambda_reg * np.maximum(0.0, ranks - k_reg + 1)      # (K,) penalty per rank
    reg_scores = sorted_probs + reg[None, :]
    cum = np.cumsum(reg_scores, axis=1)

    rank_of_y = np.zeros(n, dtype=np.int64)
    for i in range(n):
        rank_of_y[i] = int(np.where(order[i] == y[i])[0][0])

    cum_at_y = cum[np.arange(n), rank_of_y]
    contribution_at_y = reg_scores[np.arange(n), rank_of_y]
    u = rng.random(n)
    return cum_at_y - u * contribution_at_y


def build_prediction_sets_raps(
    probs: np.ndarray,
    q_hat: float,
    k_reg: int = 1,
    lambda_reg: float = 0.01,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """RAPS prediction set construction paired with raps_scores."""
    rng = rng or np.random.default_rng(0)
    n, k = probs.shape
    order = np.argsort(-probs, axis=1, kind="stable")
    sorted_probs = np.take_along_axis(probs, order, axis=1)
    ranks = np.arange(k, dtype=np.float64)
    reg = lambda_reg * np.maximum(0.0, ranks - k_reg + 1)
    reg_scores = sorted_probs + reg[None, :]
    cum = np.cumsum(reg_scores, axis=1)
    sets = np.zeros((n, k), dtype=bool)
    u = rng.random(n)
    for i in range(n):
        rank = int(np.searchsorted(cum[i], q_hat, side="left"))
        if rank == k:
            sets[i, :] = True
            continue
        keep_ranks = list(range(rank))
        include_boundary = (cum[i, rank] - u[i] * reg_scores[i, rank]) <= q_hat
        if include_boundary:
            keep_ranks.append(rank)
        if not keep_ranks:
            keep_ranks = [0]
        for r in keep_ranks:
            sets[i, order[i, r]] = True
    return sets


def evaluate_conformal_ccp(
    probs: np.ndarray,
    y: np.ndarray,
    alpha: float = 0.10,
    split_seed: int = 42,
    cal_frac: float = 0.5,
) -> dict:
    """Mondrian (class-conditional) conformal prediction with LAC score.

    Unlike standard split CP which fits one global quantile, CCP fits a
    separate quantile per true class on the calibration split. This trades
    a strictly higher marginal guarantee for a class-conditional one that
    survives on minority grades. Citation: Vovk 2003 (Mondrian taxonomy).
    """
    rng = np.random.default_rng(split_seed)
    n = len(y)
    idx = rng.permutation(n)
    n_cal = int(round(cal_frac * n))
    cal_idx, test_idx = idx[:n_cal], idx[n_cal:]

    probs_cal, y_cal = probs[cal_idx], y[cal_idx]
    probs_test, y_test = probs[test_idx], y[test_idx]
    k = probs.shape[1]

    # Per-class LAC thresholds.
    class_q = np.full(k, np.nan, dtype=np.float64)
    for c in range(k):
        mask = (y_cal == c)
        if mask.sum() == 0:
            # No cal samples of this class -- fall back to the global LAC
            # quantile so we never emit an empty threshold.
            class_q[c] = _quantile_hi(lac_scores(probs_cal, y_cal), alpha)
        else:
            scores_c = 1.0 - probs_cal[mask, c]
            class_q[c] = _quantile_hi(scores_c, alpha)

    # Include class c if probs_test[:, c] >= 1 - class_q[c].
    thresholds = 1.0 - class_q                                        # (K,)
    sets = probs_test >= thresholds[None, :]

    # Safety: never emit an empty set; if all below threshold, include argmax.
    empty = sets.sum(axis=1) == 0
    if empty.any():
        idx_fix = np.where(empty)[0]
        sets[idx_fix, probs_test[idx_fix].argmax(axis=1)] = True

    covered = sets[np.arange(len(y_test)), y_test]
    set_sizes = sets.sum(axis=1)
    ccov: list[float] = []
    for c in range(k):
        mask = (y_test == c)
        if mask.sum() == 0:
            ccov.append(float("nan"))
        else:
            ccov.append(float(covered[mask].mean()))

    return {
        "alpha": alpha,
        "score": "ccp_lac",
        "class_thresholds": class_q.tolist(),
        "n_cal": int(n_cal),
        "n_test": int(n - n_cal),
        "marginal_coverage": float(covered.mean()),
        "mean_set_size": float(set_sizes.mean()),
        "class_conditional_coverage": ccov,
        "set_size_histogram": [int((set_sizes == s).sum()) for s in range(k + 1)],
        "split_seed": split_seed,
    }


def is_contiguous(sets: np.ndarray) -> np.ndarray:
    """Per-row: is the prediction set a contiguous run of grades?

    LAC/APS/RAPS treat the K grades as an unordered label set, so they can
    return sets like {0, 3} that skip an intermediate grade. Under an ordinal
    scale such a set is semantically incoherent -- it asserts the patient is
    either healthy or severe but definitely not moderate. This measures how
    often that happens.

    An empty set is reported as contiguous (vacuously); the builders here
    always return at least one class.
    """
    n, k = sets.shape
    out = np.ones(n, dtype=bool)
    for i in range(n):
        idx = np.flatnonzero(sets[i])
        if idx.size > 1:
            out[i] = bool(idx[-1] - idx[0] + 1 == idx.size)
    return out


def _ordinal_nesting_order(probs: np.ndarray) -> np.ndarray:
    """Order in which grades enter a mode-anchored contiguous interval.

    Starting from the arg-max grade, repeatedly absorb whichever neighbour of
    the current interval carries more probability mass. This yields a strictly
    nested family of CONTIGUOUS intervals I_1 subset I_2 subset ... subset I_K
    with |I_j| = j, which is what makes the resulting conformal sets intervals
    by construction rather than by post-hoc repair.

    Returns:
        (N, K) int array; row i lists grades in the order they are absorbed.
    """
    n, k = probs.shape
    order = np.zeros((n, k), dtype=np.int64)
    for i in range(n):
        lo = hi = int(np.argmax(probs[i]))
        seq = [lo]
        while len(seq) < k:
            can_left, can_right = lo - 1 >= 0, hi + 1 <= k - 1
            # Ties break left (towards the lower grade), which is the
            # safety-conservative direction for a screening task.
            if can_left and (not can_right or probs[i, lo - 1] >= probs[i, hi + 1]):
                lo -= 1
                seq.append(lo)
            else:
                hi += 1
                seq.append(hi)
        order[i] = seq
    return order


def ocp_scores(
    probs: np.ndarray, y: np.ndarray, rng: np.random.Generator | None = None
) -> np.ndarray:
    """OCP (Ordinal Contiguous Prediction) nonconformity score.

    Structurally identical to APS, but the classes are accumulated in the
    ordinal nesting order of `_ordinal_nesting_order` instead of in
    descending-probability order. The score for (x, y) is the probability
    mass of the smallest contiguous mode-anchored interval containing y,
    randomised at the boundary exactly as in Romano (2020).

    Because the score is a fixed measurable function of (x, y) evaluated
    identically on calibration and test points, split-conformal exchangeability
    -- and therefore the 1-alpha marginal coverage guarantee -- carries over
    unchanged from APS. Contiguity is obtained for free from the nesting rule;
    it is not an extra constraint that has to be paid for in coverage.
    """
    rng = rng or np.random.default_rng(0)
    n, k = probs.shape
    order = _ordinal_nesting_order(probs)
    sorted_probs = np.take_along_axis(probs, order, axis=1)
    cum = np.cumsum(sorted_probs, axis=1)

    rank_of_y = np.zeros(n, dtype=np.int64)
    for i in range(n):
        rank_of_y[i] = int(np.where(order[i] == y[i])[0][0])

    cum_at_y = cum[np.arange(n), rank_of_y]
    p_at_y = sorted_probs[np.arange(n), rank_of_y]
    u = rng.random(n)
    return cum_at_y - u * p_at_y


def build_prediction_sets_ocp(
    probs: np.ndarray, q_hat: float, rng: np.random.Generator | None = None
) -> np.ndarray:
    """OCP prediction sets: always contiguous intervals over the grade scale.

    Mirrors `build_prediction_sets_aps` so that the constructed set is the
    q_hat sublevel set of `ocp_scores` under the same uniform draw, but walks
    the ordinal nesting order, so every returned set is an interval [a, b].
    """
    rng = rng or np.random.default_rng(0)
    n, k = probs.shape
    order = _ordinal_nesting_order(probs)
    sorted_probs = np.take_along_axis(probs, order, axis=1)
    cum = np.cumsum(sorted_probs, axis=1)

    sets = np.zeros((n, k), dtype=bool)
    u = rng.random(n)
    for i in range(n):
        rank = int(np.searchsorted(cum[i], q_hat, side="left"))
        if rank == k:
            sets[i, :] = True
            continue
        keep_ranks = list(range(rank))
        if (cum[i, rank] - u[i] * sorted_probs[i, rank]) <= q_hat:
            keep_ranks.append(rank)
        if not keep_ranks:
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
    assert score in ("lac", "aps", "raps", "ocp")
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
    elif score == "aps":
        cal_scores = aps_scores(probs_cal, y_cal, rng=rng)
        q_hat = _quantile_hi(cal_scores, alpha)
        sets = build_prediction_sets_aps(probs_test, q_hat, rng=rng)
    elif score == "raps":
        cal_scores = raps_scores(probs_cal, y_cal, k_reg=1, lambda_reg=0.01, rng=rng)
        q_hat = _quantile_hi(cal_scores, alpha)
        sets = build_prediction_sets_raps(probs_test, q_hat, k_reg=1, lambda_reg=0.01, rng=rng)
    else:  # "ocp"
        cal_scores = ocp_scores(probs_cal, y_cal, rng=rng)
        q_hat = _quantile_hi(cal_scores, alpha)
        sets = build_prediction_sets_ocp(probs_test, q_hat, rng=rng)

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
        "noncontiguous_rate": float(1.0 - is_contiguous(sets).mean()),
        "split_seed": split_seed,
    }
