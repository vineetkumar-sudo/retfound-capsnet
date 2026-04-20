"""Day 8 — IDRiD UQ + per-class figures.

Uses the saved Ordinal CapsNet IDRiD test-set predictions (with head_lengths) to
produce:
  results/figures/fig_c_idrid_uq_boxplot.{png,pdf}
  results/figures/fig_d_idrid_acc_coverage.{png,pdf}
  results/figures/fig_f_idrid_cm_sidebyside.{png,pdf}

The goal: check that the Day 5B UQ signal (prediction margin) still separates
correct from misclassified IDRiD predictions — if yes, UQ generalizes cross-
dataset; if no, UQ is an APTOS-specific artifact.

Usage:
    uv run python scripts/make_day8_figures.py
"""

from __future__ import annotations

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
IDRID_ORDINAL_DIR = Path("results/idrid/ordinal_capsnet")
IDRID_VANILLA_DIR = Path("results/idrid/capsnet_vanilla")
FIGDIR = Path("results/figures")
FIGDIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 12,
    "axes.labelsize": 13,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 11,
    "savefig.bbox": "tight",
})


def save_both(fig: plt.Figure, stem: str) -> None:
    fig.savefig(FIGDIR / f"{stem}.png", dpi=300)
    fig.savefig(FIGDIR / f"{stem}.pdf")
    plt.close(fig)


def _pool_ordinal(dir_: Path) -> dict:
    """Pool test-set arrays across 5 folds: y_true, y_pred, head_lengths."""
    files = sorted(dir_.glob("preds_fold*.npz"))
    yts, yps, hls = [], [], []
    for f in files:
        d = np.load(f)
        yts.append(d["test_y_true"])
        yps.append(d["test_y_pred"])
        hls.append(d["test_head_lengths"])
    return {
        "y_true": np.concatenate(yts),
        "y_pred": np.concatenate(yps),
        "head_lengths": np.concatenate(hls, axis=0),
    }


def _pool_vanilla(dir_: Path) -> dict | None:
    files = sorted(dir_.glob("preds_fold*.npz"))
    if not files:
        return None
    yts, yps = [], []
    for f in files:
        d = np.load(f)
        yts.append(d["test_y_true"])
        yps.append(d["test_y_pred"])
    return {"y_true": np.concatenate(yts), "y_pred": np.concatenate(yps)}


def _plot_cm(ax, cm_pct: np.ndarray, subtitle: str | None, cbar: bool) -> None:
    sns.heatmap(
        cm_pct, annot=True, fmt=".1f", cmap="Blues",
        xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=ax,
        cbar=cbar, vmin=0, vmax=100,
        annot_kws={"fontsize": 11},
        cbar_kws={"label": "% of true class"} if cbar else None,
    )
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    if subtitle:
        ax.set_title(subtitle, fontsize=12, pad=6)


def fig_f_sidebyside_cm() -> None:
    """Pooled test-set confusion matrices for vanilla vs ordinal, on IDRiD."""
    v = _pool_vanilla(IDRID_VANILLA_DIR)
    o = _pool_ordinal(IDRID_ORDINAL_DIR)
    if v is None:
        print("  Figure F skipped: vanilla IDRiD preds missing")
        return
    cm_v = confusion_matrix(v["y_true"], v["y_pred"], labels=range(5))
    cm_o = confusion_matrix(o["y_true"], o["y_pred"], labels=range(5))
    cm_v_pct = cm_v.astype(float) / cm_v.sum(axis=1, keepdims=True).clip(min=1) * 100
    cm_o_pct = cm_o.astype(float) / cm_o.sum(axis=1, keepdims=True).clip(min=1) * 100
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.2))
    _plot_cm(axes[0], cm_v_pct, "(a) Vanilla CapsNet  (IDRiD test)", cbar=False)
    _plot_cm(axes[1], cm_o_pct, "(b) Ordinal CapsNet  (IDRiD test)", cbar=True)
    fig.tight_layout()
    save_both(fig, "fig_f_idrid_cm_sidebyside")
    print("  Figure F (IDRiD CM side-by-side) saved.")


def fig_c_idrid_uq_boxplot() -> None:
    o = _pool_ordinal(IDRID_ORDINAL_DIR)
    hl = torch.from_numpy(o["head_lengths"])
    margin = prediction_margin(hl).numpy()
    correct = o["y_true"] == o["y_pred"]

    corr_v = margin[correct]; mis_v = margin[~correct]
    if not len(mis_v) or not len(corr_v):
        print("  Figure C (IDRiD) skipped: degenerate split")
        return
    _, p = mannwhitneyu(mis_v, corr_v, alternative="greater")
    p64 = float(p)

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
    ax.set_ylim(-0.03, 1.18)
    ax.plot([1, 1, 2, 2], [1.03, 1.07, 1.07, 1.03], color="black", lw=1.2)
    if p64 < 1e-300:
        ptxt = "***   (p < 1e-300)"
    elif p64 < 0.001:
        ptxt = f"***   (p = {p64:.2e})"
    elif p64 < 0.01:
        ptxt = f"**    (p = {p64:.3f})"
    elif p64 < 0.05:
        ptxt = f"*     (p = {p64:.3f})"
    else:
        ptxt = f"n.s.  (p = {p64:.3f})"
    ax.text(1.5, 1.095, f"Mann-Whitney U   {ptxt}",
            ha="center", va="bottom", fontsize=10)
    fig.tight_layout()
    save_both(fig, "fig_c_idrid_uq_boxplot")
    print(f"  Figure C (IDRiD UQ boxplot) saved  |  p = {p64:.3e}")


def fig_d_idrid_acc_coverage() -> None:
    o = _pool_ordinal(IDRID_ORDINAL_DIR)
    hl = torch.from_numpy(o["head_lengths"])
    pm = prediction_margin(hl).numpy()
    en = digit_cap_entropy(hl).numpy()
    correct = (o["y_true"] == o["y_pred"]).astype(int)
    baseline_acc = float(correct.mean())

    coverages = np.linspace(0.10, 1.0, 91)

    fig, ax = plt.subplots(figsize=(7.5, 5.0))
    for label, arr, color in [
        ("Prediction margin", pm, "#5c6bc0"),
        ("DigitCap entropy",  en, "#4c9f70"),
    ]:
        order = np.argsort(arr)
        cs = correct[order]
        accs = []
        for c in coverages:
            keep = int(round(c * len(cs)))
            accs.append(cs[:keep].mean() if keep else np.nan)
        ax.plot(coverages, accs, label=label, linewidth=2.2, color=color)

    ax.axhline(baseline_acc, linestyle="--", color="gray", linewidth=1.5,
               label=f"Baseline accuracy @ 100% coverage = {baseline_acc:.3f}")
    ax.set_xlabel("Coverage  (fraction retained, most certain first)")
    ax.set_ylabel("Accuracy on retained subset")
    ax.set_xlim(0.1, 1.02)
    ax.invert_xaxis()
    ax.grid(alpha=0.25, linestyle=":")
    ax.legend(loc="lower right", framealpha=0.95)
    fig.tight_layout()
    save_both(fig, "fig_d_idrid_acc_coverage")
    print("  Figure D (IDRiD accuracy-coverage) saved.")


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--strict", action="store_true",
                    help="Re-raise instead of skipping on the first figure error.")
    args = ap.parse_args()

    print(f"Writing IDRiD figures to {FIGDIR}/ ...")
    figures = [
        ("Figure F (IDRiD CM)",        fig_f_sidebyside_cm),
        ("Figure C (IDRiD UQ)",        fig_c_idrid_uq_boxplot),
        ("Figure D (IDRiD coverage)",  fig_d_idrid_acc_coverage),
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
