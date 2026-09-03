"""Day-13 calibration + selective-prediction analysis.

Walks every cached `preds_fold*.npz` file under a list of experiment dirs,
pools across seeds and folds, derives chain-rule 5-class probabilities from
the K-1 head probabilities, and computes:

  ECE (10 equal-mass bins) -- lower is better.
  MCE (10 equal-mass bins) -- lower is better.
  AURC + Excess-AURC       -- lower is better.
  Risk @ 80% coverage      -- lower is better.
  Risk @ 50% coverage      -- lower is better.
  Monotonicity renorm rate -- fraction of samples whose chain-rule P(y=k)
                               did not sum to 1 natively; reported as a
                               quality check on the ordinal decomposition.

Emits:
  results/calibration_table.md   paper-ready markdown
  results/calibration_table.csv  machine-readable
  results/figures/fig_reliability_diagrams.{png,pdf}
                                 4-panel reliability diagram for the APTOS
                                 head x backbone grid.

Usage:
  uv run python scripts/compute_calibration_metrics.py
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, ".")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.calibration import (
    aurc,
    chain_rule_probs,
    expected_calibration_error,
    max_confidence,
    maximum_calibration_error,
    reliability_bins,
    risk_at_coverage,
)


# ---------------------------------------------------------------------------
# Which experiments to cover
# ---------------------------------------------------------------------------
# Each entry: (display_name, dataset_label, list_of_glob_patterns). Globs
# resolve to a list of preds_fold*.npz files to pool across seeds/folds.

EXPERIMENTS = [
    ("RETFound x Ordinal CapsNet",    "APTOS",        ["results/ordinal_capsnet/seed*/preds_fold*.npz"]),
    ("RETFound x MLP + K-1 sigmoid",  "APTOS",        ["results/mlp_ordinal/seed*/preds_fold*.npz"]),
    ("DINOv2  x Ordinal CapsNet",     "APTOS",        ["results/ordinal_capsnet_dinov2/seed*/preds_fold*.npz"]),
    ("DINOv2  x MLP + K-1 sigmoid",   "APTOS",        ["results/mlp_ordinal_dinov2/seed*/preds_fold*.npz"]),
    ("RETFound x Ordinal CapsNet + LoRA", "APTOS",    ["results/lora_ordinal_capsnet/seed*/preds_fold*.npz"]),
    ("RETFound x Ordinal CapsNet",    "Messidor-2",   ["results/messidor2/ordinal_capsnet/preds_fold*.npz"]),
    ("RETFound x MLP + K-1 sigmoid",  "Messidor-2",   ["results/messidor2/mlp_k1_sigmoid/preds_fold*.npz"]),
    ("DINOv2  x Ordinal CapsNet",     "Messidor-2",   ["results/messidor2_dinov2/ordinal_capsnet/preds_fold*.npz"]),
    ("DINOv2  x MLP + K-1 sigmoid",   "Messidor-2",   ["results/messidor2_dinov2/mlp_k1_sigmoid/preds_fold*.npz"]),
]


def pool_npz(globs: list[str]) -> dict[str, np.ndarray] | None:
    """Concatenate preds arrays across all npz files matching the globs.

    Keys pooled: y_true, y_pred, head_lengths. (head_probs is derived from
    head_lengths[:,:,1] so we don't rely on it being stored.)
    """
    paths: list[Path] = []
    for g in globs:
        paths.extend(sorted(Path(".").glob(g)))
    if not paths:
        return None
    yts, yps, hls = [], [], []
    for p in paths:
        d = np.load(p)
        yts.append(d["y_true"])
        yps.append(d["y_pred"])
        hls.append(d["head_lengths"])
    return {
        "y_true": np.concatenate(yts).astype(np.int64),
        "y_pred": np.concatenate(yps).astype(np.int64),
        "head_lengths": np.concatenate(hls, axis=0).astype(np.float32),
        "_n_files": len(paths),
    }


def compute_metrics(pool: dict[str, np.ndarray], num_classes: int = 5) -> dict:
    """Compute every calibration + selective-prediction metric for one pool."""
    y_true = pool["y_true"]
    head_probs = pool["head_lengths"][:, :, 1]                  # (N, K-1)
    probs, renorm_rate = chain_rule_probs(head_probs, num_classes=num_classes)

    # The paper's inference rule is the K-1 thresholded count, not argmax(P),
    # so we take accuracy against that rule and use P(y = predicted_grade) as
    # the per-sample confidence. This is the honest ECE for the classifier we
    # actually ship: how calibrated is the model's probability in its own
    # K-1 decision, not in the arg-max of an implied 5-way posterior.
    y_pred = pool["y_pred"]
    correct = (y_pred == y_true).astype(np.int64)
    accuracy = float(correct.mean())
    confs = probs[np.arange(len(y_pred)), y_pred]

    ece = expected_calibration_error(confs, correct, n_bins=10, mass=True)
    mce = maximum_calibration_error(confs, correct, n_bins=10, mass=True)
    bins = reliability_bins(confs, correct, n_bins=10, mass=True)
    aurc_val, excess_aurc = aurc(confs, correct)
    risk80 = risk_at_coverage(confs, correct, 0.80)
    risk50 = risk_at_coverage(confs, correct, 0.50)

    # Prediction margin as alternative UQ selector (higher margin = more
    # certain; so we negate when using as "confidence" input to AURC).
    top2 = np.partition(probs, -2, axis=1)[:, -2:]
    pred_margin = top2[:, 1] - top2[:, 0]                       # in [0, 1]
    aurc_pm, excess_pm = aurc(pred_margin, correct)
    risk80_pm = risk_at_coverage(pred_margin, correct, 0.80)
    risk50_pm = risk_at_coverage(pred_margin, correct, 0.50)

    return {
        "n_samples": int(len(y_true)),
        "n_files_pooled": pool["_n_files"],
        "accuracy": accuracy,
        "ece": ece,
        "mce": mce,
        "renorm_rate": renorm_rate,
        "aurc_maxprob": aurc_val,
        "excess_aurc_maxprob": excess_aurc,
        "risk_at_80_maxprob": risk80,
        "risk_at_50_maxprob": risk50,
        "aurc_margin": aurc_pm,
        "excess_aurc_margin": excess_pm,
        "risk_at_80_margin": risk80_pm,
        "risk_at_50_margin": risk50_pm,
        "reliability_bins": bins,
    }


# ---------------------------------------------------------------------------
# Output: markdown, CSV, reliability figure
# ---------------------------------------------------------------------------

def write_table(rows: list[tuple[str, str, dict]], out_md: Path, out_csv: Path) -> None:
    """Emit both a paper-ready .md and a .csv with identical content."""
    headers = [
        "Dataset", "Model",
        "Acc", "ECE", "MCE",
        "AURC(maxp)", "Excess-AURC(maxp)",
        "Risk@80(maxp)", "Risk@50(maxp)",
        "AURC(margin)", "Excess-AURC(margin)",
        "Renorm%",
    ]

    lines: list[str] = []
    lines.append("# Calibration + selective-prediction analysis")
    lines.append("")
    lines.append("All rows pool per-sample predictions across every fold and (where available) seed under an experiment directory. "
                 "ECE / MCE use 10 equal-mass bins over max chain-rule probability. AURC integrates risk over coverage when ranking "
                 "samples by confidence (lower is better); Excess-AURC subtracts the oracle AURC for the same error count, making "
                 "the number comparable across datasets with different base error rates. Two confidence signals are compared: "
                 "max chain-rule probability (standard) and prediction margin (our capsule-native UQ signal).")
    lines.append("")
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("|" + "|".join(["---"] * len(headers)) + "|")

    csv_rows = [headers]
    for dataset, model, m in rows:
        row = [
            dataset, model,
            f"{m['accuracy']:.4f}",
            f"{m['ece']:.4f}",
            f"{m['mce']:.4f}",
            f"{m['aurc_maxprob']:.4f}",
            f"{m['excess_aurc_maxprob']:.4f}",
            f"{m['risk_at_80_maxprob']:.4f}",
            f"{m['risk_at_50_maxprob']:.4f}",
            f"{m['aurc_margin']:.4f}",
            f"{m['excess_aurc_margin']:.4f}",
            f"{100 * m['renorm_rate']:.2f}",
        ]
        lines.append("| " + " | ".join(row) + " |")
        csv_rows.append(row)

    out_md.write_text("\n".join(lines) + "\n")
    with out_csv.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(csv_rows)


def write_reliability_figure(
    rows: list[tuple[str, str, dict]],
    aptos_only: bool,
    out_png: Path,
    out_pdf: Path,
    dataset: str = "APTOS",
) -> None:
    """4-panel reliability diagram for the head x backbone grid on `dataset`."""
    selected = [(mdl, m) for ds, mdl, m in rows if ds == dataset and "LoRA" not in mdl]
    if not selected:
        return
    # Reviewer M1: with sharex/sharey matplotlib suppresses the tick VALUES on
    # the inner panels, so two of the four sub-plots showed no axis numbers.
    # Every panel now carries explicit ticks and labels.
    fig, axes = plt.subplots(2, 2, figsize=(9.5, 8.5))
    axes = axes.ravel()
    ticks = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    for ax, (name, m) in zip(axes, selected):
        bins = m["reliability_bins"]
        xs = [b["mean_conf"] for b in bins if b["count"] > 0]
        ys = [b["mean_acc"] for b in bins if b["count"] > 0]
        ws = [b["weight"]    for b in bins if b["count"] > 0]
        ax.plot([0, 1], [0, 1], "k--", alpha=0.5, linewidth=1)
        ax.scatter(xs, ys, s=[200 * w for w in ws], alpha=0.7,
                   color="#3B5EAA", edgecolor="black", linewidth=0.5)
        for x, y, w in zip(xs, ys, ws):
            ax.plot([x, x], [x, y], color="#C26A6A", alpha=0.5, linewidth=1)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.set_xticks(ticks); ax.set_yticks(ticks)
        ax.set_xticklabels([f"{t:.1f}" for t in ticks], fontsize=8)
        ax.set_yticklabels([f"{t:.1f}" for t in ticks], fontsize=8)
        ax.set_title(f"{name}\nECE={m['ece']:.3f}  MCE={m['mce']:.3f}", fontsize=10)
        ax.set_xlabel("Mean predicted confidence", fontsize=9)
        ax.set_ylabel("Empirical accuracy", fontsize=9)
        ax.grid(True, alpha=0.3)
    fig.suptitle(f"Reliability diagrams on {dataset}\n"
                 f"(10 equal-mass bins; dashed = perfect calibration; bubble size = bin weight)",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150)
    fig.savefig(out_pdf)
    plt.close(fig)


def write_risk_coverage_figure(
    rows: list[tuple[str, str, dict]],
    out_png: Path,
    out_pdf: Path,
    dataset: str = "APTOS",
) -> None:
    """Risk-coverage curves on `dataset` (max-prob confidence selector)."""
    selected = [(mdl, m) for ds, mdl, m in rows if ds == dataset]
    if not selected:
        return
    # Rebuild the curves from bins -- instead we need per-sample confs. We
    # stored them in the rebuilt pool, but the summary dict above does not
    # carry them. Re-read the pools. Cheap, keeps the module testable.
    fig, ax = plt.subplots(figsize=(8.0, 5.5))
    colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(selected)))
    for (name, m), color in zip(selected, colors):
        # Reconstruct the curve from the saved reliability bins is lossy;
        # recompute from pooled preds instead.
        pool = _cached_pool[name + "||" + dataset]
        head_probs = pool["head_lengths"][:, :, 1]
        probs, _ = chain_rule_probs(head_probs)
        confs, _ = max_confidence(probs)
        correct = (pool["y_pred"] == pool["y_true"]).astype(np.int64)
        order = np.argsort(-confs, kind="stable")
        cum_err = np.cumsum(1 - correct[order])
        n = len(confs)
        cov = np.arange(1, n + 1) / n
        risk = cum_err / np.arange(1, n + 1)
        ax.plot(cov, risk, label=f"{name}  AURC={m['aurc_maxprob']:.3f}",
                color=color, linewidth=1.8)
    ax.set_xlabel("Coverage (fraction of samples retained, highest confidence first)")
    ax.set_ylabel("Risk on retained set (1 - accuracy)")
    ax.set_title(f"Risk-coverage curves on {dataset}\n(selector: max chain-rule probability)")
    ax.set_xlim(0, 1); ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    fig.savefig(out_pdf)
    plt.close(fig)


# Global cache so write_risk_coverage_figure can re-use the pools the main
# loop already built (avoids re-reading the npz files from disk).
_cached_pool: dict[str, dict] = {}


# ---------------------------------------------------------------------------

def main() -> None:
    out_root = Path("results")
    fig_root = Path("results/figures")
    out_root.mkdir(parents=True, exist_ok=True)
    fig_root.mkdir(parents=True, exist_ok=True)

    rows: list[tuple[str, str, dict]] = []
    for name, dataset, globs in EXPERIMENTS:
        pool = pool_npz(globs)
        if pool is None:
            print(f"  SKIP  {dataset:<12s} {name:<40s}  (no matching preds files)")
            continue
        m = compute_metrics(pool)
        rows.append((dataset, name, m))
        _cached_pool[name + "||" + dataset] = pool
        print(f"  OK    {dataset:<12s} {name:<40s}  "
              f"N={m['n_samples']:>5d}  Acc={m['accuracy']:.4f}  "
              f"ECE={m['ece']:.4f}  MCE={m['mce']:.4f}  "
              f"AURC(maxp)={m['aurc_maxprob']:.4f}  "
              f"AURC(margin)={m['aurc_margin']:.4f}  "
              f"renorm={100*m['renorm_rate']:.1f}%")

    write_table(rows, out_root / "calibration_table.md", out_root / "calibration_table.csv")
    write_reliability_figure(
        rows, aptos_only=True,
        out_png=fig_root / "fig_reliability_aptos.png",
        out_pdf=fig_root / "fig_reliability_aptos.pdf",
        dataset="APTOS",
    )
    write_reliability_figure(
        rows, aptos_only=False,
        out_png=fig_root / "fig_reliability_messidor2.png",
        out_pdf=fig_root / "fig_reliability_messidor2.pdf",
        dataset="Messidor-2",
    )
    write_risk_coverage_figure(
        rows,
        out_png=fig_root / "fig_risk_coverage_aptos.png",
        out_pdf=fig_root / "fig_risk_coverage_aptos.pdf",
        dataset="APTOS",
    )
    write_risk_coverage_figure(
        rows,
        out_png=fig_root / "fig_risk_coverage_messidor2.png",
        out_pdf=fig_root / "fig_risk_coverage_messidor2.pdf",
        dataset="Messidor-2",
    )
    # Also dump the raw dict to JSON so later scripts can re-use without
    # re-pooling. Reliability-bin lists are preserved for diagram reuse.
    dump = [{"dataset": d, "model": n, **m} for d, n, m in rows]
    (out_root / "calibration_metrics.json").write_text(json.dumps(dump, indent=2))
    print(f"\nWrote {out_root/'calibration_table.md'} ({len(rows)} rows)")
    print(f"Wrote {fig_root/'fig_reliability_aptos.pdf'}")
    print(f"Wrote {fig_root/'fig_reliability_messidor2.pdf'}")
    print(f"Wrote {fig_root/'fig_risk_coverage_aptos.pdf'}")
    print(f"Wrote {fig_root/'fig_risk_coverage_messidor2.pdf'}")


if __name__ == "__main__":
    main()
