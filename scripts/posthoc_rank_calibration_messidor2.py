"""Post-hoc rank calibration for APTOS → Messidor-2 cross-dataset eval.

The frozen cross-dataset QWK collapses to ~0.01 because mean P(y > 0)
drops from ~0.51 on APTOS to ~0.09 on Messidor-2 — the model predicts
Grade 0 for 99.8 % of samples. This is a feature-space shift, not a head
bug: the ordinal ranking of severity scores is still useful, it's just
the 0.5 threshold that's miscalibrated for the new domain.

Rank calibration corrects that by re-assigning grades so the predicted
marginal distribution matches a reference distribution while preserving
the rank order of the continuous severity score.

Two variants are reported:
  (a) APTOS-distribution calibration (no target-domain peeking) —
      ranks are bucketed using the APTOS *train* label proportions.
      This is the realistic / honest version: you only need to know the
      source domain's class prior.
  (b) Oracle-distribution calibration (sanity-check upper bound) —
      ranks are bucketed using the Messidor-2 ground-truth marginal.
      This measures how much of the QWK loss is pure distribution-shift
      vs genuine feature-space failure.

For the ordinal head, continuous severity = sum_k P(y > k) (i.e. the
expected grade under the chain-rule probability model), pooled across
the 5 CV folds via fold-mean of head_probs.

Outputs (all under `results/cross_dataset/aptos_to_messidor2/calibrated/`):
  summary.json            — QWK / Acc / F1 / MAE / binary metrics per variant
  preds_aptos_prior.npz   — calibrated 5-class preds using APTOS prior
  preds_oracle.npz        — calibrated 5-class preds using Messidor-2 GT prior

Usage:
    uv run python scripts/posthoc_rank_calibration_messidor2.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, ".")

import numpy as np
from sklearn.metrics import roc_auc_score

from src.evaluate import compute_all_metrics


APTOS_LABELS = Path("data/aptos/features/train_labels.npy")
CROSS_DIR = Path("results/cross_dataset/aptos_to_messidor2/ordinal_capsnet")
OUT_DIR = Path("results/cross_dataset/aptos_to_messidor2/calibrated")


def class_prior(labels: np.ndarray, n_classes: int = 5) -> np.ndarray:
    """Return the marginal class distribution as a float array summing to 1."""
    counts = np.bincount(labels, minlength=n_classes).astype(np.float64)
    return counts / counts.sum()


def rank_calibrate(score: np.ndarray, prior: np.ndarray) -> np.ndarray:
    """Assign grades so the predicted marginal matches `prior`.

    Samples are sorted by `score` ascending and split into buckets whose
    sizes are proportional to `prior`. Ties in the score are broken by
    argsort's stable ordering (insertion-order preserving for equal keys).

    Args:
        score:  (N,) continuous severity score (higher = more severe)
        prior:  (K,) target class marginal (must sum to ~1)

    Returns:
        (N,) int array of calibrated grades in [0, K-1].
    """
    n = len(score)
    order = np.argsort(score, kind="stable")
    # Cumulative bucket boundaries — floor to integer counts, round last.
    cumulative = np.cumsum(prior) * n
    boundaries = np.concatenate(([0], np.round(cumulative).astype(int)))
    boundaries[-1] = n  # absorb any rounding drift into the top bucket
    calibrated = np.empty(n, dtype=np.int64)
    for grade, (lo, hi) in enumerate(zip(boundaries[:-1], boundaries[1:])):
        calibrated[order[lo:hi]] = grade
    return calibrated


def pool_head_probs(cross_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fold-average head_probs across the 5 per-fold npz files.

    Returns (head_probs_mean, y_true_5class, y_true_binary).
    """
    files = sorted(cross_dir.glob("preds_fold*.npz"))
    assert len(files) == 5, f"expected 5 per-fold npz, got {len(files)}"
    hps = []
    for f in files:
        d = np.load(f)
        hps.append(d["head_probs"])
    hp = np.stack(hps, axis=0).mean(axis=0)  # (1744, 4)
    d0 = np.load(files[0])
    return hp, d0["y_true_5class"], d0["y_true_binary"]


def binary_metrics(y_true: np.ndarray, score: np.ndarray,
                   threshold: float = 0.5) -> dict:
    """Sensitivity / Specificity / AUC / F1 / accuracy at `threshold`."""
    y_true = np.asarray(y_true).astype(int)
    pred = (score >= threshold).astype(int)
    tp = int(((pred == 1) & (y_true == 1)).sum())
    tn = int(((pred == 0) & (y_true == 0)).sum())
    fp = int(((pred == 1) & (y_true == 0)).sum())
    fn = int(((pred == 0) & (y_true == 1)).sum())
    sens = tp / max(tp + fn, 1)
    spec = tn / max(tn + fp, 1)
    prec = tp / max(tp + fp, 1)
    f1 = 2 * prec * sens / max(prec + sens, 1e-12)
    try:
        auc = float(roc_auc_score(y_true, score))
    except ValueError:
        auc = float("nan")
    return {
        "threshold": float(threshold),
        "sensitivity": float(sens),
        "specificity": float(spec),
        "auc": auc,
        "f1": float(f1),
        "accuracy": float((pred == y_true).mean()),
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    hp, y_5, y_bin = pool_head_probs(CROSS_DIR)
    severity = hp.sum(axis=1)  # E[grade] ∈ [0, 4]
    uncal_pred = (hp >= 0.5).sum(axis=1)  # original chain-rule grade

    aptos_prior = class_prior(np.load(APTOS_LABELS))
    oracle_prior = class_prior(y_5)

    print("Predicted marginal BEFORE calibration:")
    _p = np.bincount(uncal_pred, minlength=5) / len(uncal_pred)
    print(f"  {_p.round(4).tolist()}")
    print(f"APTOS train prior:      {aptos_prior.round(4).tolist()}")
    print(f"Messidor-2 GT prior:    {oracle_prior.round(4).tolist()}\n")

    m_uncal = compute_all_metrics(y_5, uncal_pred, num_classes=5)

    pred_aptos = rank_calibrate(severity, aptos_prior)
    m_aptos = compute_all_metrics(y_5, pred_aptos, num_classes=5)

    pred_oracle = rank_calibrate(severity, oracle_prior)
    m_oracle = compute_all_metrics(y_5, pred_oracle, num_classes=5)

    # Binary (referable): score is severity (continuous), threshold is rank-
    # calibrated so the expected positive rate matches the source prior.
    # For APTOS, referable = grade >= 2 has prior p = 1 - aptos_prior[:2].sum().
    aptos_referable_rate = 1.0 - aptos_prior[:2].sum()
    oracle_referable_rate = 1.0 - oracle_prior[:2].sum()
    thr_aptos = float(np.quantile(severity, 1.0 - aptos_referable_rate))
    thr_oracle = float(np.quantile(severity, 1.0 - oracle_referable_rate))

    b_uncal = binary_metrics(y_bin, severity, threshold=0.5)  # the collapsing threshold
    b_aptos = binary_metrics(y_bin, severity, threshold=thr_aptos)
    b_oracle = binary_metrics(y_bin, severity, threshold=thr_oracle)

    print("5-class (QWK / Acc / F1 / MAE):")
    fmt = "  {lbl:<28s}  QWK={q:.4f}  Acc={a:.4f}  F1={f:.4f}  MAE={m:.4f}"
    print(fmt.format(lbl="uncalibrated (baseline)",  q=m_uncal["qwk"],  a=m_uncal["accuracy"],  f=m_uncal["macro_f1"],  m=m_uncal["mae"]))
    print(fmt.format(lbl="APTOS-prior calibrated",   q=m_aptos["qwk"],  a=m_aptos["accuracy"],  f=m_aptos["macro_f1"],  m=m_aptos["mae"]))
    print(fmt.format(lbl="oracle (M2-GT) calibrated", q=m_oracle["qwk"], a=m_oracle["accuracy"], f=m_oracle["macro_f1"], m=m_oracle["mae"]))

    print("\nBinary (referable = grade >= 2):")
    b_fmt = "  {lbl:<28s}  Sens={s:.3f}  Spec={sp:.3f}  F1={f:.3f}  AUC={auc:.3f}  thr={t:.3f}"
    print(b_fmt.format(lbl="uncalibrated (thr=0.5)", s=b_uncal["sensitivity"], sp=b_uncal["specificity"], f=b_uncal["f1"], auc=b_uncal["auc"], t=b_uncal["threshold"]))
    print(b_fmt.format(lbl="APTOS-prior calibrated", s=b_aptos["sensitivity"], sp=b_aptos["specificity"], f=b_aptos["f1"], auc=b_aptos["auc"], t=b_aptos["threshold"]))
    print(b_fmt.format(lbl="oracle (M2-GT) calibrated", s=b_oracle["sensitivity"], sp=b_oracle["specificity"], f=b_oracle["f1"], auc=b_oracle["auc"], t=b_oracle["threshold"]))

    (OUT_DIR / "summary.json").write_text(json.dumps({
        "source": {
            "cross_dir": str(CROSS_DIR),
            "n_samples": int(len(y_5)),
            "method": "Ordinal CapsNet (frozen RETFound)",
            "severity_score": "sum_k P(y > k) from chain-rule head_probs (fold-averaged)",
        },
        "priors": {
            "aptos_train_5class": aptos_prior.round(6).tolist(),
            "messidor2_gt_5class": oracle_prior.round(6).tolist(),
            "aptos_referable_rate": round(float(aptos_referable_rate), 6),
            "messidor2_gt_referable_rate": round(float(oracle_referable_rate), 6),
        },
        "predicted_marginal_uncalibrated": (
            (np.bincount(uncal_pred, minlength=5) / len(uncal_pred)).round(6).tolist()
        ),
        "five_class": {
            "uncalibrated": {k: float(v) for k, v in m_uncal.items()
                             if k in ("qwk", "accuracy", "macro_f1", "mae")},
            "aptos_prior_calibrated": {k: float(v) for k, v in m_aptos.items()
                                       if k in ("qwk", "accuracy", "macro_f1", "mae")},
            "oracle_calibrated": {k: float(v) for k, v in m_oracle.items()
                                  if k in ("qwk", "accuracy", "macro_f1", "mae")},
        },
        "binary": {
            "uncalibrated_thr_0_5": b_uncal,
            "aptos_prior_calibrated": b_aptos,
            "oracle_calibrated": b_oracle,
        },
    }, indent=2))

    np.savez_compressed(OUT_DIR / "preds_aptos_prior.npz",
                        y_true_5class=y_5, y_true_binary=y_bin,
                        y_pred_uncalibrated=uncal_pred,
                        y_pred_calibrated=pred_aptos,
                        severity=severity.astype(np.float32))
    np.savez_compressed(OUT_DIR / "preds_oracle.npz",
                        y_true_5class=y_5, y_true_binary=y_bin,
                        y_pred_uncalibrated=uncal_pred,
                        y_pred_calibrated=pred_oracle,
                        severity=severity.astype(np.float32))
    print(f"\nWrote {OUT_DIR}/summary.json + 2 preds npz")


if __name__ == "__main__":
    main()
