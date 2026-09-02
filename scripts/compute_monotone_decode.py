"""Rank-monotonicity as a constrained projection vs. clip-and-renormalise.

Reviewer comment 5 objected that the decoding in Section III-D is "an
unconstrained thresholding process, where probabilities that are not monotonic
are just truncated and then normalized", and asked for the rank-monotonicity
constraint to be stated as an optimisation problem. This script implements and
ablates that.

Three decodings of the K-1 head outputs h_k = P(y > k):

  A  product + clip + renormalise      (what the paper did)
       P(y=0) = 1 - h_1;  P(y=k) = h_k * (1 - h_{k+1});  P(y=K-1) = h_{K-1}
       then clip at 0 and divide by the row sum.

  B  difference + clip + renormalise
       P(y=k) = h_k - h_{k+1}, then clip and renormalise.

  C  monotone projection + difference   (proposed)
       h_hat = argmin_q 0.5||q - h||^2  s.t.  1 >= q_1 >= ... >= q_{K-1} >= 0
       solved exactly by PAV in O(K), then the difference decode. Under a
       monotone h_hat the result is non-negative and sums to exactly 1, so no
       clipping and no renormalisation occur at all.

Two diagnostics motivate C. First, the product decode does not normalise even
for a perfectly monotone h -- e.g. h = (0.5, 0.4, 0.3, 0.2) sums to 1.52 -- so
the paper's Renorm% column (100% on nearly every row) measures a decoding
artefact, not ordinal inconsistency as its caption states. Second, the true
violation rate of the CONTINUOUS heads is far higher than the thresholded
`count_non_monotonic` rate the paper reports, because the latter only counts
contradictions among the hard h > 0.5 decisions.

This ablation also answers reviewer comment R2-2 ("how does clipping negative
probabilities affect calibration?"), since B clips and C never needs to.

Outputs results/monotone_decode_table.{md,csv} and
results/monotone_decode_metrics.json.

Usage:
    uv run python scripts/compute_monotone_decode.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.calibration import (
    aurc,
    chain_rule_probs,
    expected_calibration_error,
    max_confidence,
    maximum_calibration_error,
    monotone_projection,
)
from src.evaluate import compute_all_metrics

EXPERIMENTS = [
    ("RETFound x Ordinal CapsNet",        "APTOS",      ["results/ordinal_capsnet/seed*/preds_fold*.npz"]),
    ("RETFound x MLP + K-1 sigmoid",      "APTOS",      ["results/mlp_ordinal/seed*/preds_fold*.npz"]),
    ("DINOv2  x Ordinal CapsNet",         "APTOS",      ["results/ordinal_capsnet_dinov2/seed*/preds_fold*.npz"]),
    ("DINOv2  x MLP + K-1 sigmoid",       "APTOS",      ["results/mlp_ordinal_dinov2/seed*/preds_fold*.npz"]),
    ("RETFound x Ordinal CapsNet + LoRA", "APTOS",      ["results/lora_ordinal_capsnet/seed*/preds_fold*.npz"]),
    ("RETFound x Ordinal CapsNet",        "Messidor-2", ["results/messidor2/ordinal_capsnet/preds_fold*.npz"]),
    ("RETFound x MLP + K-1 sigmoid",      "Messidor-2", ["results/messidor2/mlp_k1_sigmoid/preds_fold*.npz"]),
    ("DINOv2  x Ordinal CapsNet",         "Messidor-2", ["results/messidor2_dinov2/ordinal_capsnet/preds_fold*.npz"]),
    ("DINOv2  x MLP + K-1 sigmoid",       "Messidor-2", ["results/messidor2_dinov2/mlp_k1_sigmoid/preds_fold*.npz"]),
]

VARIANTS = [
    ("A product+clip+renorm", dict(decode="product", project=False)),
    ("B difference+clip+renorm", dict(decode="difference", project=False)),
    ("C projection+difference", dict(decode="difference", project=True)),
]


def pool(globs: list[str]) -> dict | None:
    paths: list[Path] = []
    for g in globs:
        paths.extend(sorted(Path(".").glob(g)))
    if not paths:
        return None
    yts, hls = [], []
    for p in paths:
        d = np.load(p)
        yts.append(d["y_true"])
        hls.append(d["head_lengths"])
    return {
        "y_true": np.concatenate(yts).astype(np.int64),
        "h": np.concatenate(hls, axis=0).astype(np.float64)[:, :, 1],
    }


def eval_variant(h: np.ndarray, y: np.ndarray, kw: dict) -> dict:
    probs, renorm = chain_rule_probs(h, **kw)
    conf, pred = max_confidence(probs)
    correct = (pred == y).astype(float)
    m = compute_all_metrics(y, pred)
    a, _ = aurc(conf, correct)
    return {
        "qwk": m["qwk"],
        "accuracy": m["accuracy"],
        "macro_f1": m["macro_f1"],
        "ece": expected_calibration_error(conf, correct, n_bins=10, mass=True),
        "mce": maximum_calibration_error(conf, correct, n_bins=10, mass=True),
        "aurc": a,
        "renorm_rate": renorm,
    }


def main() -> None:
    rows: list[dict] = []
    for name, dataset, globs in EXPERIMENTS:
        p = pool(globs)
        if p is None:
            print(f"  SKIP  {dataset:<12s} {name}")
            continue
        h, y = p["h"], p["y_true"]
        _, viol = monotone_projection(h)
        rec: dict = {
            "dataset": dataset, "model": name, "n": int(len(y)),
            "continuous_violation_rate": viol,
        }
        for label, kw in VARIANTS:
            rec[label] = eval_variant(h, y, kw)
        rows.append(rec)
        a, c = rec["A product+clip+renorm"], rec["C projection+difference"]
        print(f"  {dataset:<11s} {name:<38s} viol={100 * viol:5.1f}%  "
              f"ECE {a['ece']:.4f} -> {c['ece']:.4f}  "
              f"QWK {a['qwk']:.4f} -> {c['qwk']:.4f}")

    out = Path("results")
    (out / "monotone_decode_metrics.json").write_text(json.dumps(rows, indent=2))

    md = [
        "# Rank-monotonicity: constrained projection vs. clip-and-renormalise",
        "",
        "A = product decode + clip + renormalise (as submitted).",
        "B = cumulative-difference decode + clip + renormalise.",
        "C = monotone projection (PAV) + cumulative-difference decode (proposed);",
        "needs no clipping and no renormalisation -- rows sum to exactly 1.",
        "",
        "`viol%` is the fraction of samples whose CONTINUOUS heads violate",
        "monotonicity, i.e. the fraction the projection actually moves. This is",
        "distinct from the paper's `count_non_monotonic`, which only counts",
        "contradictions among the thresholded h > 0.5 decisions.",
        "",
        "| Dataset | Model | viol% | ECE A | ECE B | ECE C | QWK A | QWK B | QWK C "
        "| Acc A | Acc C | Renorm% A | Renorm% C |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        a, b, c = (r["A product+clip+renorm"], r["B difference+clip+renorm"],
                   r["C projection+difference"])
        md.append(
            f"| {r['dataset']} | {r['model']} | {100 * r['continuous_violation_rate']:.1f}% "
            f"| {a['ece']:.4f} | {b['ece']:.4f} | **{c['ece']:.4f}** "
            f"| {a['qwk']:.4f} | {b['qwk']:.4f} | **{c['qwk']:.4f}** "
            f"| {a['accuracy']:.4f} | **{c['accuracy']:.4f}** "
            f"| {100 * a['renorm_rate']:.1f}% | {100 * c['renorm_rate']:.1f}% |"
        )
    (out / "monotone_decode_table.md").write_text("\n".join(md) + "\n")

    hdr = ["dataset", "model", "n", "continuous_violation_rate", "variant",
           "qwk", "accuracy", "macro_f1", "ece", "mce", "aurc", "renorm_rate"]
    csv = [",".join(hdr)]
    for r in rows:
        for label, _ in VARIANTS:
            v = r[label]
            csv.append(",".join([
                r["dataset"], r["model"], str(r["n"]),
                f"{r['continuous_violation_rate']:.6f}", label,
                f"{v['qwk']:.6f}", f"{v['accuracy']:.6f}", f"{v['macro_f1']:.6f}",
                f"{v['ece']:.6f}", f"{v['mce']:.6f}", f"{v['aurc']:.6f}",
                f"{v['renorm_rate']:.6f}",
            ]))
    (out / "monotone_decode_table.csv").write_text("\n".join(csv) + "\n")
    print("\nWrote results/monotone_decode_table.{md,csv} + "
          "results/monotone_decode_metrics.json")


if __name__ == "__main__":
    main()
