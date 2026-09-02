"""OCP (Ordinal Contiguous Prediction) vs LAC / APS / RAPS across the grid.

Answers two things the submitted paper asserted but could not support.

1. The non-contiguous-set rate. Section V-D claimed specific rates for LAC,
   APS and RAPS, but no code in this repository ever computed them. This
   script measures them directly (`src.conformal.is_contiguous`).

2. An ordinal-aware conformal construction. Reviewer comment 9 asked for a
   method whose prediction sets are contiguous intervals on the grade scale.
   `src.conformal.ocp_scores` accumulates probability mass in a mode-anchored
   contiguous nesting order rather than in descending-probability order, so
   every set is an interval [a, b] by construction. Because the score is a
   fixed measurable function of (x, y) applied identically to calibration and
   test points, the split-conformal marginal-coverage guarantee is inherited
   from APS unchanged -- contiguity costs nothing in validity, and this script
   quantifies what (if anything) it costs in set size.

Outputs results/ordinal_conformal_table.{md,csv} and
results/ordinal_conformal_metrics.json.

Usage:
    uv run python scripts/compute_ordinal_conformal.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.calibration import chain_rule_probs
from src.conformal import evaluate_conformal

# Same nine cells, same glob patterns and split seeds as
# scripts/compute_conformal_metrics.py, so the two tables are comparable.
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

ALPHAS = [0.10, 0.05]
SCORES = ["lac", "aps", "raps", "ocp"]
SPLIT_SEEDS = [42, 123, 456, 789, 1011]


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
        "head_lengths": np.concatenate(hls, axis=0).astype(np.float32),
        "n_files": len(paths),
    }


def _agg(vals: list[float]) -> tuple[float, float]:
    a = np.asarray(vals, dtype=float)
    return float(a.mean()), float(a.std())


def eval_config(p: dict) -> dict:
    probs, _ = chain_rule_probs(p["head_lengths"][:, :, 1])
    y = p["y_true"]
    out: dict = {"n_total": int(len(y)), "n_files_pooled": int(p["n_files"])}

    for alpha in ALPHAS:
        for score in SCORES:
            cov, size, ncg, worst = [], [], [], []
            for seed in SPLIT_SEEDS:
                r = evaluate_conformal(probs, y, alpha=alpha, score=score,
                                       split_seed=seed)
                cov.append(r["marginal_coverage"])
                size.append(r["mean_set_size"])
                ncg.append(r["noncontiguous_rate"])
                cc = [c for c in r["class_conditional_coverage"]
                      if c is not None and not np.isnan(c)]
                worst.append(min(cc) if cc else float("nan"))
            key = f"alpha{alpha:.2f}_{score}"
            out[key] = {
                "coverage_mean": _agg(cov)[0], "coverage_std": _agg(cov)[1],
                "set_size_mean": _agg(size)[0], "set_size_std": _agg(size)[1],
                "noncontiguous_rate_mean": _agg(ncg)[0],
                "noncontiguous_rate_std": _agg(ncg)[1],
                "worst_class_coverage_mean": _agg(worst)[0],
                "worst_class_coverage_std": _agg(worst)[1],
            }
    return out


def main() -> None:
    rows: list[tuple[str, str, dict]] = []
    for name, dataset, globs in EXPERIMENTS:
        p = pool(globs)
        if p is None:
            print(f"  SKIP  {dataset:<12s} {name:<40s} (no preds)")
            continue
        r = eval_config(p)
        rows.append((dataset, name, r))
        a = r["alpha0.10_ocp"]
        b = r["alpha0.10_aps"]
        print(f"  {dataset:<11s} {name:<38s} "
              f"OCP |S|={a['set_size_mean']:.2f} cov={a['coverage_mean']:.3f} "
              f"nc={100 * a['noncontiguous_rate_mean']:.1f}%  |  "
              f"APS |S|={b['set_size_mean']:.2f} nc={100 * b['noncontiguous_rate_mean']:.1f}%")

    out_root = Path("results")
    (out_root / "ordinal_conformal_metrics.json").write_text(
        json.dumps([{"dataset": d, "model": m, **r} for d, m, r in rows], indent=2)
    )

    # --- Markdown: non-contiguity + OCP-vs-APS efficiency, alpha = 0.10 -----
    md = [
        "# Ordinal contiguous conformal prediction (OCP)",
        "",
        "Five random 50:50 cal/test splits per cell (seeds 42/123/456/789/1011);",
        "mean +/- std across splits. `non-contig%` is the fraction of prediction",
        "sets that skip an intermediate grade (e.g. {0, 3}), which is incoherent",
        "under an ordinal scale. OCP is 0% by construction.",
        "",
        "## alpha = 0.10 (target coverage 90%)",
        "",
        "| Dataset | Model | LAC non-contig% | APS non-contig% | RAPS non-contig% "
        "| APS cov | APS \\|S\\| | OCP cov | OCP \\|S\\| | OCP worst-class |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for d, m, r in rows:
        lac, aps, raps, ocp = (r["alpha0.10_lac"], r["alpha0.10_aps"],
                               r["alpha0.10_raps"], r["alpha0.10_ocp"])
        md.append(
            f"| {d} | {m} | {100 * lac['noncontiguous_rate_mean']:.2f}% "
            f"| {100 * aps['noncontiguous_rate_mean']:.2f}% "
            f"| {100 * raps['noncontiguous_rate_mean']:.2f}% "
            f"| {aps['coverage_mean']:.4f} | {aps['set_size_mean']:.3f} "
            f"| {ocp['coverage_mean']:.4f} | {ocp['set_size_mean']:.3f} "
            f"| {ocp['worst_class_coverage_mean']:.4f} |"
        )
    md += ["", "## alpha = 0.05 (target coverage 95%)", "",
           "| Dataset | Model | APS non-contig% | RAPS non-contig% | APS \\|S\\| "
           "| OCP cov | OCP \\|S\\| |", "|---|---|---|---|---|---|---|"]
    for d, m, r in rows:
        aps, raps, ocp = (r["alpha0.05_aps"], r["alpha0.05_raps"], r["alpha0.05_ocp"])
        md.append(
            f"| {d} | {m} | {100 * aps['noncontiguous_rate_mean']:.2f}% "
            f"| {100 * raps['noncontiguous_rate_mean']:.2f}% "
            f"| {aps['set_size_mean']:.3f} "
            f"| {ocp['coverage_mean']:.4f} | {ocp['set_size_mean']:.3f} |"
        )
    (out_root / "ordinal_conformal_table.md").write_text("\n".join(md) + "\n")

    hdr = ["dataset", "model", "alpha", "score", "coverage_mean", "coverage_std",
           "set_size_mean", "set_size_std", "noncontiguous_rate_mean",
           "worst_class_coverage_mean"]
    csv = [",".join(hdr)]
    for d, m, r in rows:
        for alpha in ALPHAS:
            for score in SCORES:
                v = r[f"alpha{alpha:.2f}_{score}"]
                csv.append(",".join([
                    d, m, f"{alpha:.2f}", score,
                    f"{v['coverage_mean']:.6f}", f"{v['coverage_std']:.6f}",
                    f"{v['set_size_mean']:.6f}", f"{v['set_size_std']:.6f}",
                    f"{v['noncontiguous_rate_mean']:.6f}",
                    f"{v['worst_class_coverage_mean']:.6f}",
                ]))
    (out_root / "ordinal_conformal_table.csv").write_text("\n".join(csv) + "\n")

    print("\nWrote results/ordinal_conformal_table.{md,csv} + "
          "results/ordinal_conformal_metrics.json")


if __name__ == "__main__":
    main()
