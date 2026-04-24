"""Day-13B split-conformal evaluation across the head x backbone grid.

Reuses the pooled preds from compute_calibration_metrics.py (same 9 configs,
same pool = all folds x seeds per config). For each config we run split
conformal with both LAC and APS nonconformity scores, at alpha = 0.10 and
alpha = 0.05, across 5 random cal/test splits, and report the mean +/- std
of marginal coverage and mean set size. Class-conditional coverage is
reported for the single reproducible split (seed=42) since averaging a K-wide
vector over splits inflates result volume with little extra insight.

Emits:
  results/conformal_table.md        paper-ready markdown
  results/conformal_table.csv       machine-readable
  results/conformal_metrics.json    structured per-config detail

Usage:
  uv run python scripts/compute_conformal_metrics.py
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, ".")

import numpy as np

from src.calibration import chain_rule_probs
from src.conformal import (
    _quantile_hi,
    build_prediction_sets_raps,
    evaluate_conformal,
    evaluate_conformal_ccp,
    raps_scores,
)


EXPERIMENTS = [
    ("RETFound x Ordinal CapsNet",    "APTOS",      ["results/ordinal_capsnet/seed*/preds_fold*.npz"]),
    ("RETFound x MLP + K-1 sigmoid",  "APTOS",      ["results/mlp_ordinal/seed*/preds_fold*.npz"]),
    ("DINOv2  x Ordinal CapsNet",     "APTOS",      ["results/ordinal_capsnet_dinov2/seed*/preds_fold*.npz"]),
    ("DINOv2  x MLP + K-1 sigmoid",   "APTOS",      ["results/mlp_ordinal_dinov2/seed*/preds_fold*.npz"]),
    ("RETFound x Ordinal CapsNet + LoRA", "APTOS",  ["results/lora_ordinal_capsnet/seed*/preds_fold*.npz"]),
    ("RETFound x Ordinal CapsNet",    "Messidor-2", ["results/messidor2/ordinal_capsnet/preds_fold*.npz"]),
    ("RETFound x MLP + K-1 sigmoid",  "Messidor-2", ["results/messidor2/mlp_k1_sigmoid/preds_fold*.npz"]),
    ("DINOv2  x Ordinal CapsNet",     "Messidor-2", ["results/messidor2_dinov2/ordinal_capsnet/preds_fold*.npz"]),
    ("DINOv2  x MLP + K-1 sigmoid",   "Messidor-2", ["results/messidor2_dinov2/mlp_k1_sigmoid/preds_fold*.npz"]),
]

ALPHAS = [0.10, 0.05]
SCORES = ["lac", "aps", "raps"]
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
    }


def eval_config(p: dict) -> dict:
    probs, _ = chain_rule_probs(p["head_lengths"][:, :, 1])
    y = p["y_true"]

    out: dict = {"n_total": int(len(y))}
    # LAC / APS / RAPS (single global threshold)
    for alpha in ALPHAS:
        for score in SCORES:
            covs, sizes, q_hats = [], [], []
            ccov_reference = None
            size_hist_reference = None
            for seed in SPLIT_SEEDS:
                r = evaluate_conformal(probs, y, alpha=alpha, score=score, split_seed=seed)
                covs.append(r["marginal_coverage"])
                sizes.append(r["mean_set_size"])
                q_hats.append(r["q_hat"])
                if seed == SPLIT_SEEDS[0]:
                    ccov_reference = r["class_conditional_coverage"]
                    size_hist_reference = r["set_size_histogram"]
            out[f"alpha{alpha:.2f}_{score}"] = {
                "marginal_coverage_mean": float(np.mean(covs)),
                "marginal_coverage_std": float(np.std(covs)),
                "mean_set_size_mean": float(np.mean(sizes)),
                "mean_set_size_std": float(np.std(sizes)),
                "q_hat_mean": float(np.mean(q_hats)),
                "class_conditional_coverage_seed0": ccov_reference,
                "set_size_histogram_seed0": size_hist_reference,
            }

    # CCP (Mondrian, per-class quantile) — addresses the class-conditional
    # collapse; separate loop because it does not use a single q_hat.
    for alpha in ALPHAS:
        covs, sizes = [], []
        ccov_reference = None
        size_hist_reference = None
        for seed in SPLIT_SEEDS:
            r = evaluate_conformal_ccp(probs, y, alpha=alpha, split_seed=seed)
            covs.append(r["marginal_coverage"])
            sizes.append(r["mean_set_size"])
            if seed == SPLIT_SEEDS[0]:
                ccov_reference = r["class_conditional_coverage"]
                size_hist_reference = r["set_size_histogram"]
        out[f"alpha{alpha:.2f}_ccp"] = {
            "marginal_coverage_mean": float(np.mean(covs)),
            "marginal_coverage_std": float(np.std(covs)),
            "mean_set_size_mean": float(np.mean(sizes)),
            "mean_set_size_std": float(np.std(sizes)),
            "class_conditional_coverage_seed0": ccov_reference,
            "set_size_histogram_seed0": size_hist_reference,
        }
    return out


def main() -> None:
    out_root = Path("results")
    out_root.mkdir(parents=True, exist_ok=True)

    rows: list[tuple[str, str, dict]] = []
    for name, dataset, globs in EXPERIMENTS:
        p = pool(globs)
        if p is None:
            print(f"  SKIP  {dataset:<12s} {name:<40s}  (no preds)")
            continue
        r = eval_config(p)
        rows.append((dataset, name, r))
        a10_lac = r["alpha0.10_lac"]
        a10_aps = r["alpha0.10_aps"]
        a10_raps = r["alpha0.10_raps"]
        a10_ccp = r["alpha0.10_ccp"]
        # Worst-class from APS (matches the original headline in the paper).
        worst_aps = min(c for c in a10_aps["class_conditional_coverage_seed0"]
                        if c is not None and not np.isnan(c))
        worst_ccp = min(c for c in a10_ccp["class_conditional_coverage_seed0"]
                        if c is not None and not np.isnan(c))
        print(f"  {dataset:<11s} {name:<38s}  "
              f"LAC |S|={a10_lac['mean_set_size_mean']:.2f}  "
              f"APS |S|={a10_aps['mean_set_size_mean']:.2f}  "
              f"RAPS |S|={a10_raps['mean_set_size_mean']:.2f}  "
              f"CCP |S|={a10_ccp['mean_set_size_mean']:.2f}  "
              f"worst-class APS={worst_aps:.2f} -> CCP={worst_ccp:.2f}")

    # ---- Markdown table ----
    md: list[str] = []
    md.append("# Split conformal prediction — paper-ready table")
    md.append("")
    md.append(f"Five random cal/test 50:50 splits per config; mean ± std reported. Each row is pooled across all folds × seeds for that config (same pool as `compute_calibration_metrics.py`). Empty cells where no preds exist.")
    md.append("")
    md.append("## Coverage and set-size at α = 0.10 (target 90% coverage)")
    md.append("")
    md.append("Four nonconformity scores compared: LAC (Sadinle 2019), APS "
              "(Romano 2020), RAPS (Angelopoulos 2021, $k_\\text{reg}{=}1$, "
              "$\\lambda{=}0.01$), and CCP (Mondrian per-class quantile, "
              "Vovk 2003). LAC/APS/RAPS are all marginal-coverage methods; "
              "CCP is class-conditional. $|S|$ is mean prediction set size "
              "(smaller at equal coverage is better).")
    md.append("")
    md.append("| Dataset | Model | LAC cov | LAC \\|S\\| | APS cov | APS \\|S\\| | RAPS cov | RAPS \\|S\\| | CCP cov | CCP \\|S\\| |")
    md.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    csv_rows = [["dataset", "model",
                 "LAC_0.10_coverage_mean", "LAC_0.10_coverage_std",
                 "LAC_0.10_setsize_mean",  "LAC_0.10_setsize_std",
                 "APS_0.10_coverage_mean", "APS_0.10_coverage_std",
                 "APS_0.10_setsize_mean",  "APS_0.10_setsize_std",
                 "LAC_0.05_coverage_mean", "LAC_0.05_setsize_mean",
                 "APS_0.05_coverage_mean", "APS_0.05_setsize_mean"]]
    for dataset, name, r in rows:
        L10 = r["alpha0.10_lac"]; A10 = r["alpha0.10_aps"]
        R10 = r["alpha0.10_raps"]; C10 = r["alpha0.10_ccp"]
        L05 = r["alpha0.05_lac"]; A05 = r["alpha0.05_aps"]
        md.append(f"| {dataset} | {name} | "
                  f"{L10['marginal_coverage_mean']:.3f} | {L10['mean_set_size_mean']:.2f} | "
                  f"{A10['marginal_coverage_mean']:.3f} | {A10['mean_set_size_mean']:.2f} | "
                  f"{R10['marginal_coverage_mean']:.3f} | {R10['mean_set_size_mean']:.2f} | "
                  f"{C10['marginal_coverage_mean']:.3f} | {C10['mean_set_size_mean']:.2f} |")
        csv_rows.append([dataset, name,
                         f"{L10['marginal_coverage_mean']:.4f}", f"{L10['marginal_coverage_std']:.4f}",
                         f"{L10['mean_set_size_mean']:.4f}",     f"{L10['mean_set_size_std']:.4f}",
                         f"{A10['marginal_coverage_mean']:.4f}", f"{A10['marginal_coverage_std']:.4f}",
                         f"{A10['mean_set_size_mean']:.4f}",     f"{A10['mean_set_size_std']:.4f}",
                         f"{L05['marginal_coverage_mean']:.4f}", f"{L05['mean_set_size_mean']:.4f}",
                         f"{A05['marginal_coverage_mean']:.4f}", f"{A05['mean_set_size_mean']:.4f}",
                         f"{R10['marginal_coverage_mean']:.4f}", f"{R10['mean_set_size_mean']:.4f}",
                         f"{C10['marginal_coverage_mean']:.4f}", f"{C10['mean_set_size_mean']:.4f}"])
    md.append("")
    md.append("## Coverage and set-size at α = 0.05 (target 95% coverage)")
    md.append("")
    md.append("| Dataset | Model | LAC cov | LAC mean\\|S\\| | APS cov | APS mean\\|S\\| |")
    md.append("|---|---|---:|---:|---:|---:|")
    for dataset, name, r in rows:
        L = r["alpha0.05_lac"]; A = r["alpha0.05_aps"]
        md.append(f"| {dataset} | {name} | "
                  f"{L['marginal_coverage_mean']:.3f} ± {L['marginal_coverage_std']:.3f} | "
                  f"{L['mean_set_size_mean']:.2f} ± {L['mean_set_size_std']:.2f} | "
                  f"{A['marginal_coverage_mean']:.3f} ± {A['marginal_coverage_std']:.3f} | "
                  f"{A['mean_set_size_mean']:.2f} ± {A['mean_set_size_std']:.2f} |")

    md.append("")
    md.append("## Class-conditional coverage at α = 0.10 (single-split seed 42)")
    md.append("")
    md.append("Worst-per-class coverage is the honest deployment metric: the "
              "marginal guarantee from LAC/APS/RAPS does not transfer to per-class "
              "coverage. CCP (Mondrian, per-class quantile) is the canonical fix "
              "and is the closest non-RC3P baseline achievable on MPS without "
              "new training. Values are per-class coverage; `min` is the worst "
              "class; `|S|` is the mean CCP set size at this alpha.")
    md.append("")
    md.append("### APS (single global threshold)")
    md.append("")
    md.append("| Dataset | Model | C0 | C1 | C2 | C3 | C4 | min |")
    md.append("|---|---|---:|---:|---:|---:|---:|---:|")
    for dataset, name, r in rows:
        ccov = r["alpha0.10_aps"]["class_conditional_coverage_seed0"]
        valid = [c for c in ccov if c is not None and not (isinstance(c, float) and np.isnan(c))]
        md.append(f"| {dataset} | {name} | "
                  + " | ".join(f"{c:.3f}" if c is not None and not np.isnan(c) else "—" for c in ccov)
                  + f" | {min(valid):.3f} |")
    md.append("")
    md.append("### CCP (Mondrian, per-class quantile)")
    md.append("")
    md.append("| Dataset | Model | C0 | C1 | C2 | C3 | C4 | min | CCP \\|S\\| |")
    md.append("|---|---|---:|---:|---:|---:|---:|---:|---:|")
    for dataset, name, r in rows:
        ccov = r["alpha0.10_ccp"]["class_conditional_coverage_seed0"]
        size = r["alpha0.10_ccp"]["mean_set_size_mean"]
        valid = [c for c in ccov if c is not None and not (isinstance(c, float) and np.isnan(c))]
        md.append(f"| {dataset} | {name} | "
                  + " | ".join(f"{c:.3f}" if c is not None and not np.isnan(c) else "—" for c in ccov)
                  + f" | {min(valid):.3f} | {size:.2f} |")

    (out_root / "conformal_table.md").write_text("\n".join(md) + "\n")
    with (out_root / "conformal_table.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(csv_rows)
    (out_root / "conformal_metrics.json").write_text(json.dumps(
        [{"dataset": d, "model": n, **r} for d, n, r in rows], indent=2))

    print(f"\nWrote {out_root/'conformal_table.md'}")
    print(f"Wrote {out_root/'conformal_table.csv'}")


if __name__ == "__main__":
    main()
