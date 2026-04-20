"""Day 6 — Generate the 5 publication-quality APTOS figures.

Outputs to `results/figures/` as both 300-dpi PNG and vector PDF:
  fig_a_cm_vanilla.{png,pdf}         Vanilla CapsNet confusion matrix (Day 2)
  fig_b_cm_ordinal.{png,pdf}         Ordinal CapsNet confusion matrix (Day 3, seed 42)
  fig_ab_cm_sidebyside.{png,pdf}     1x2 side-by-side of A and B
  fig_c_uq_boxplot.{png,pdf}         Prediction-margin boxplot: correct vs misclassified
  fig_d_acc_coverage.{png,pdf}       Accuracy-coverage curves for all 3 UQ methods
  fig_e_training_curve.{png,pdf}     Per-epoch val QWK, mean +/- std band (Ordinal vs Vanilla)

All data sources are files on disk produced by earlier scripts; this script does
no training.

Usage:
    uv run python scripts/make_day6_figures.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, ".")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from scipy.stats import mannwhitneyu
from sklearn.metrics import confusion_matrix

from src.uncertainty import digit_cap_entropy, prediction_margin


CLASS_NAMES = ["No DR", "Mild", "Moderate", "Severe", "PDR"]
FIGDIR = Path("results/figures")
FIGDIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.size": 12,
    "axes.labelsize": 13,
    "axes.titlesize": 14,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 11,
    "savefig.bbox": "tight",
})


def save_both(fig: plt.Figure, stem: str) -> None:
    """Save figure as 300dpi PNG and vector PDF."""
    fig.savefig(FIGDIR / f"{stem}.png", dpi=300)
    fig.savefig(FIGDIR / f"{stem}.pdf")
    plt.close(fig)


def _ordinal_preds_dir() -> Path:
    """Return the directory holding Ordinal CapsNet seed-42 per-fold preds.
    Prefers the Day 6 seed42/ subdir; falls back to the legacy top-level layout."""
    seeded = Path("results/ordinal_capsnet/seed42")
    if any(seeded.glob("preds_fold*.npz")):
        return seeded
    return Path("results/ordinal_capsnet")


def pool_fold_npz(dir_: Path, keys: tuple[str, ...] = ("y_true", "y_pred")) -> dict:
    """Concatenate per-fold arrays across preds_fold*.npz files in `dir_`."""
    files = sorted(dir_.glob("preds_fold*.npz"))
    if not files:
        raise FileNotFoundError(f"No preds_fold*.npz in {dir_}")
    out: dict[str, list[np.ndarray]] = {k: [] for k in keys}
    for f in files:
        d = np.load(f)
        for k in keys:
            if k in d.files:
                out[k].append(d[k])
    return {k: np.concatenate(v) for k, v in out.items() if v}


# ---------------------------------------------------------------------------
# Figures A, B, AB
# ---------------------------------------------------------------------------

def _plot_cm(ax, cm_pct: np.ndarray, subtitle: str | None = None, cbar: bool = True) -> None:
    """Confusion-matrix heatmap.

    `subtitle` is only used for the side-by-side (a) / (b) labels; no top-level
    title is ever set — the paper's LaTeX caption owns all titling.
    """
    sns.heatmap(
        cm_pct, annot=True, fmt=".1f", cmap="Blues",
        xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
        ax=ax, cbar=cbar, vmin=0, vmax=100,
        annot_kws={"fontsize": 11},
        cbar_kws={"label": "% of true class"} if cbar else None,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    if subtitle:
        ax.set_title(subtitle, fontsize=12, pad=6)


def fig_a_vanilla_cm() -> None:
    d = pool_fold_npz(Path("results/capsnet"))
    cm = confusion_matrix(d["y_true"], d["y_pred"], labels=range(5))
    cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100
    fig, ax = plt.subplots(figsize=(6.0, 5.0))
    _plot_cm(ax, cm_pct)
    fig.tight_layout()
    save_both(fig, "fig_a_cm_vanilla")
    print("  Figure A saved.")


def fig_b_ordinal_cm() -> None:
    d = pool_fold_npz(_ordinal_preds_dir())
    cm = confusion_matrix(d["y_true"], d["y_pred"], labels=range(5))
    cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100
    fig, ax = plt.subplots(figsize=(6.0, 5.0))
    _plot_cm(ax, cm_pct)
    fig.tight_layout()
    save_both(fig, "fig_b_cm_ordinal")
    print("  Figure B saved.")


def fig_ab_sidebyside() -> None:
    dv = pool_fold_npz(Path("results/capsnet"))
    do = pool_fold_npz(_ordinal_preds_dir())
    cm_v = confusion_matrix(dv["y_true"], dv["y_pred"], labels=range(5))
    cm_o = confusion_matrix(do["y_true"], do["y_pred"], labels=range(5))
    cm_v_pct = cm_v.astype(float) / cm_v.sum(axis=1, keepdims=True) * 100
    cm_o_pct = cm_o.astype(float) / cm_o.sum(axis=1, keepdims=True) * 100

    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.2))
    _plot_cm(axes[0], cm_v_pct, "(a) Vanilla CapsNet", cbar=False)
    _plot_cm(axes[1], cm_o_pct, "(b) Ordinal CapsNet", cbar=True)
    fig.tight_layout()
    save_both(fig, "fig_ab_cm_sidebyside")
    print("  Figure A+B side-by-side saved.")


# ---------------------------------------------------------------------------
# Figures C, D  (UQ analysis on Ordinal CapsNet, seed 42 preds)
# ---------------------------------------------------------------------------

def _load_uq_arrays(dir_: Path) -> dict:
    """Load pooled per-sample UQ signals from preds_fold*.npz.

    Returns y_true, y_pred, prediction_margin, digit_entropy. Routing variance
    is only included when `routing_variance` is present in every npz file —
    older preds dumps predate that column and fall back to a None sentinel.
    """
    files = sorted(dir_.glob("preds_fold*.npz"))
    if not files:
        raise FileNotFoundError(f"No preds_fold*.npz in {dir_}")

    yts, yps, hls, rvs = [], [], [], []
    have_rv = True
    for f in files:
        d = np.load(f)
        yts.append(d["y_true"])
        yps.append(d["y_pred"])
        hls.append(d["head_lengths"])
        if "routing_variance" in d.files:
            rvs.append(d["routing_variance"])
        else:
            have_rv = False

    y_true = np.concatenate(yts)
    y_pred = np.concatenate(yps)
    head_lengths = torch.from_numpy(np.concatenate(hls, axis=0))
    out = {
        "y_true": y_true, "y_pred": y_pred,
        "prediction_margin": prediction_margin(head_lengths).numpy(),
        "digit_entropy": digit_cap_entropy(head_lengths).numpy(),
    }
    if have_rv and rvs:
        out["routing_variance"] = np.concatenate(rvs)
    return out


def fig_c_uq_boxplot() -> None:
    uq = _load_uq_arrays(_ordinal_preds_dir())
    correct = uq["y_true"] == uq["y_pred"]
    pm = uq["prediction_margin"]

    corr_v = pm[correct]; mis_v = pm[~correct]
    u, p = mannwhitneyu(mis_v, corr_v, alternative="greater") if len(mis_v) and len(corr_v) else (0.0, 1.0)

    fig, ax = plt.subplots(figsize=(6.5, 5.2))
    bp = ax.boxplot(
        [corr_v, mis_v],
        tick_labels=[f"Correct\n(n={len(corr_v)})", f"Misclassified\n(n={len(mis_v)})"],
        patch_artist=True, widths=0.55, showfliers=False,
    )
    bp["boxes"][0].set_facecolor("#4c9f70")
    bp["boxes"][1].set_facecolor("#c44e52")
    for b in bp["boxes"]:
        b.set_alpha(0.7); b.set_edgecolor("black")
    ax.set_ylabel("Prediction margin  (1 − [top1 − top2])")
    # No matplotlib title — LaTeX caption owns all titling.
    ax.set_ylim(-0.03, 1.18)
    ax.plot([1, 1, 2, 2], [1.03, 1.07, 1.07, 1.03], color="black", lw=1.2)
    p64 = float(p)  # scipy may return float32, where 1e-300 underflows to 0
    if p64 < 1e-300:
        ptxt = "***   (p < 1e-300)"
    else:
        ptxt = f"***   (p = {p64:.2e})"
    ax.text(1.5, 1.095, f"Mann-Whitney U   {ptxt}",
            ha="center", va="bottom", fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    save_both(fig, "fig_c_uq_boxplot")
    print(f"  Figure C saved (p-value = {p:.3e}).")


def fig_d_acc_coverage() -> None:
    uq = _load_uq_arrays(_ordinal_preds_dir())
    correct = (uq["y_true"] == uq["y_pred"]).astype(int)
    baseline_acc = float(correct.mean())

    fig, ax = plt.subplots(figsize=(7.5, 5.0))

    coverages = np.linspace(0.10, 1.0, 91)
    curve_specs = [
        ("Prediction margin", uq["prediction_margin"], "#5c6bc0"),
        ("DigitCap entropy",  uq["digit_entropy"],    "#4c9f70"),
    ]
    if "routing_variance" in uq:
        curve_specs.append(("Routing variance",  uq["routing_variance"], "#c44e52"))
    else:
        print("  Figure D: routing_variance missing from preds npz — rerun ordinal "
              "seed 42 with the patched script to populate.")

    for label, arr, color in curve_specs:
        # Reject highest-uncertainty samples first; accuracy on the retained portion
        order = np.argsort(arr)       # low -> high uncertainty
        correct_sorted = correct[order]
        accs = []
        n = len(correct_sorted)
        for c in coverages:
            keep = int(round(c * n))
            accs.append(correct_sorted[:keep].mean() if keep else np.nan)
        ax.plot(coverages, accs, label=label, linewidth=2.2, color=color)

    ax.axhline(baseline_acc, linestyle="--", color="gray", linewidth=1.5,
               label=f"Baseline accuracy @ 100% coverage = {baseline_acc:.3f}")
    ax.set_xlabel("Coverage  (fraction of samples retained, most certain first)")
    ax.set_ylabel("Accuracy on retained subset")
    # No title — LaTeX caption owns titling.
    ax.set_xlim(0.1, 1.02)
    ax.invert_xaxis()
    ax.grid(alpha=0.25, linestyle=":")
    ax.legend(loc="lower right", framealpha=0.95)
    fig.tight_layout()
    save_both(fig, "fig_d_acc_coverage")
    print("  Figure D saved.")


# ---------------------------------------------------------------------------
# Figure E — training curve
# ---------------------------------------------------------------------------

def _fold_histories_from_summary(summary_path: Path, val_key: str = "qwk") -> list[list[dict]]:
    """Return per-fold histories.

    Search order for `fold_histories.json`:
      1. Next to `summary_path` itself
      2. In `seed42/` under the same parent (new Day 6 multi-seed layout)
    Fallback: embedded `history` in per_fold entries (older runs; usually empty).
    """
    candidates = [
        summary_path.parent / "fold_histories.json",
        summary_path.parent / "seed42" / "fold_histories.json",
    ]
    for fh_path in candidates:
        if fh_path.exists():
            d = json.load(fh_path.open())
            fhs = d.get("fold_histories") or []
            if fhs:
                return fhs
    s = json.load(summary_path.open())
    per_fold = s.get("per_fold") or []
    histories = []
    for pf in per_fold:
        h = pf.get("history")
        if isinstance(h, list) and h:
            histories.append(h)
    return histories


def fig_e_training_curve() -> None:
    """Ordinal vs Vanilla val QWK across epochs, mean ± std band."""
    ord_hist = _fold_histories_from_summary(Path("results/ordinal_capsnet/summary.json"))
    van_hist = _fold_histories_from_summary(Path("results/capsnet/summary.json"))

    fig, ax = plt.subplots(figsize=(8, 5))

    def _plot(hist_list, label, color, best_epoch_key="best_epoch"):
        if not hist_list:
            print(f"  (training curve skipped for {label} — no per-epoch history in summary.json)")
            return None
        max_len = max(len(h) for h in hist_list)
        curves = np.full((len(hist_list), max_len), np.nan)
        for i, h in enumerate(hist_list):
            for j, row in enumerate(h):
                curves[i, j] = row.get("qwk", np.nan)
        mean = np.nanmean(curves, axis=0)
        std = np.nanstd(curves, axis=0)
        xs = np.arange(1, max_len + 1)
        ax.plot(xs, mean, color=color, linewidth=2.2, label=label)
        ax.fill_between(xs, mean - std, mean + std, color=color, alpha=0.2)
        return int(np.nanargmax(mean) + 1)

    ord_best = _plot(ord_hist, "Ordinal CapsNet (5-fold mean ± std)", "#2a6f97")
    _plot(van_hist, "Vanilla CapsNet (5-fold mean)", "#c44e52")

    if ord_best is not None:
        ax.axvline(ord_best, linestyle="--", color="gray", linewidth=1.3,
                   label=f"Ordinal best mean epoch = {ord_best}")

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation QWK")
    # No title — LaTeX caption owns titling.
    ax.grid(alpha=0.25, linestyle=":")
    ax.legend(loc="lower right", framealpha=0.95)
    fig.tight_layout()
    save_both(fig, "fig_e_training_curve")
    print("  Figure E saved.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--strict", action="store_true",
                    help="Re-raise the first figure-generation error instead of skipping. "
                         "Use this for paper-generation runs where every figure must land.")
    args = ap.parse_args()

    print(f"Writing figures to {FIGDIR}/ ...")
    figures = [
        ("Figure A",        fig_a_vanilla_cm),
        ("Figure B",        fig_b_ordinal_cm),
        ("Figure AB",       fig_ab_sidebyside),
        ("Figure C",        fig_c_uq_boxplot),
        ("Figure D",        fig_d_acc_coverage),
        ("Figure E",        fig_e_training_curve),
    ]
    failed: list[str] = []
    for name, fn in figures:
        try:
            fn()
        except Exception as e:
            if args.strict:
                raise
            print(f"  {name} skipped: {e}")
            failed.append(name)
    if failed:
        print(f"Done with {len(failed)} failure(s): {failed}. "
              "Rerun with --strict to see the traceback.")
    else:
        print("Done.")


if __name__ == "__main__":
    main()
