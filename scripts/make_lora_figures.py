"""LoRA companions to the Day 6 APTOS figures.

Outputs (300 dpi PNG + vector PDF in `results/figures/`):
  fig_b_cm_ordinal_lora.{png,pdf}        — LoRA Ordinal CM (pooled 5-fold)
  fig_b_frozen_vs_lora_cm.{png,pdf}      — frozen vs LoRA side-by-side
  fig_c_uq_boxplot_lora.{png,pdf}        — LoRA prediction margin correct-vs-mis
  fig_d_acc_coverage_lora.{png,pdf}      — 3-curve coverage with routing variance

Pairs with `results/figures/fig_{b,c,d}_*.png` from Day 6 (frozen ordinal) so
the paper can present a frozen-vs-LoRA comparison.

Usage:
    uv run python scripts/make_lora_figures.py [--strict]
"""

from __future__ import annotations

import argparse
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
LORA_DIR = Path("results/lora_ordinal_capsnet")
FROZEN_DIR = Path("results/ordinal_capsnet/seed42")  # Day 6 seed-42 frozen Ordinal preds
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


def _pool(dir_: Path, keys: tuple[str, ...]) -> dict:
    files = sorted(dir_.glob("preds_fold*.npz"))
    if not files:
        raise FileNotFoundError(f"No preds_fold*.npz in {dir_}")
    out: dict[str, list[np.ndarray]] = {k: [] for k in keys}
    for f in files:
        d = np.load(f)
        for k in keys:
            if k in d.files:
                out[k].append(d[k])
    return {k: np.concatenate(v, axis=0) for k, v in out.items() if v}


def _pool_lora_routing_variance() -> np.ndarray | None:
    """Stitch per-fold routing_variance_fold{N}.npy in fold order (matches preds concat)."""
    files = sorted(LORA_DIR.glob("routing_variance_fold*.npy"))
    if not files:
        return None
    arrs = [np.load(f) for f in files]
    return np.concatenate(arrs, axis=0)


# ---------------------------------------------------------------------------
# Fig B' / B-comparison: LoRA CM + frozen-vs-LoRA side-by-side
# ---------------------------------------------------------------------------

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


def fig_b_lora_cm() -> None:
    d = _pool(LORA_DIR, ("y_true", "y_pred"))
    cm = confusion_matrix(d["y_true"], d["y_pred"], labels=range(5))
    cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1) * 100
    fig, ax = plt.subplots(figsize=(6.0, 5.0))
    _plot_cm(ax, cm_pct, subtitle=None, cbar=True)
    fig.tight_layout()
    save_both(fig, "fig_b_cm_ordinal_lora")
    print("  Figure B' (LoRA CM) saved.")


def fig_b_frozen_vs_lora() -> None:
    f = _pool(FROZEN_DIR, ("y_true", "y_pred"))
    l = _pool(LORA_DIR, ("y_true", "y_pred"))
    cm_f = confusion_matrix(f["y_true"], f["y_pred"], labels=range(5))
    cm_l = confusion_matrix(l["y_true"], l["y_pred"], labels=range(5))
    cm_f_pct = cm_f.astype(float) / cm_f.sum(axis=1, keepdims=True).clip(min=1) * 100
    cm_l_pct = cm_l.astype(float) / cm_l.sum(axis=1, keepdims=True).clip(min=1) * 100

    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.2))
    _plot_cm(axes[0], cm_f_pct, "(a) Ordinal CapsNet  (frozen backbone)", cbar=False)
    _plot_cm(axes[1], cm_l_pct, "(b) Ordinal CapsNet  (LoRA fine-tune)", cbar=True)
    fig.tight_layout()
    save_both(fig, "fig_b_frozen_vs_lora_cm")
    print("  Figure B-compare (frozen vs LoRA CM) saved.")


# ---------------------------------------------------------------------------
# Fig C': LoRA UQ boxplot — prediction margin, correct vs mis
# ---------------------------------------------------------------------------

def fig_c_lora_uq_boxplot() -> None:
    d = _pool(LORA_DIR, ("y_true", "y_pred", "head_lengths"))
    hl = torch.from_numpy(d["head_lengths"])
    pm = prediction_margin(hl).numpy()
    correct = d["y_true"] == d["y_pred"]
    corr_v, mis_v = pm[correct], pm[~correct]
    if not len(corr_v) or not len(mis_v):
        print("  Figure C' skipped: degenerate split")
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
    if p64 < 1e-300:     ptxt = "***   (p < 1e-300)"
    elif p64 < 0.001:    ptxt = f"***   (p = {p64:.2e})"
    elif p64 < 0.01:     ptxt = f"**    (p = {p64:.3f})"
    elif p64 < 0.05:     ptxt = f"*     (p = {p64:.3f})"
    else:                ptxt = f"n.s.  (p = {p64:.3f})"
    ax.text(1.5, 1.095, f"Mann-Whitney U   {ptxt}",
            ha="center", va="bottom", fontsize=10)
    fig.tight_layout()
    save_both(fig, "fig_c_uq_boxplot_lora")
    print(f"  Figure C' (LoRA UQ boxplot) saved  |  p = {p64:.3e}")


# ---------------------------------------------------------------------------
# Fig D': 3-curve accuracy-coverage
# ---------------------------------------------------------------------------

def fig_d_lora_acc_coverage() -> None:
    d = _pool(LORA_DIR, ("y_true", "y_pred", "head_lengths"))
    hl = torch.from_numpy(d["head_lengths"])
    pm = prediction_margin(hl).numpy()
    en = digit_cap_entropy(hl).numpy()
    rv = _pool_lora_routing_variance()

    correct = (d["y_true"] == d["y_pred"]).astype(int)
    baseline_acc = float(correct.mean())

    coverages = np.linspace(0.10, 1.0, 91)
    specs = [
        ("Prediction margin", pm, "#5c6bc0"),
        ("DigitCap entropy",  en, "#4c9f70"),
    ]
    if rv is not None and len(rv) == len(correct):
        specs.append(("Routing variance", rv, "#c44e52"))
    else:
        print("  Figure D': routing_variance missing or misaligned — "
              "run scripts/compute_lora_routing_variance.py to populate.")

    fig, ax = plt.subplots(figsize=(7.5, 5.0))
    for label, arr, color in specs:
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
    save_both(fig, "fig_d_acc_coverage_lora")
    print("  Figure D' (LoRA accuracy-coverage) saved.")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--strict", action="store_true")
    args = ap.parse_args()

    print(f"Writing LoRA figures to {FIGDIR}/ ...")
    figs = [
        ("Figure B' (LoRA CM)",            fig_b_lora_cm),
        ("Figure B-compare (frozen vs LoRA)", fig_b_frozen_vs_lora),
        ("Figure C' (LoRA UQ boxplot)",    fig_c_lora_uq_boxplot),
        ("Figure D' (LoRA accuracy-coverage)", fig_d_lora_acc_coverage),
    ]
    failed: list[str] = []
    for name, fn in figs:
        try:
            fn()
        except Exception as e:
            if args.strict:
                raise
            print(f"  {name} skipped: {e}")
            failed.append(name)
    if failed:
        print(f"Done with {len(failed)} failure(s): {failed}. Rerun with --strict for traceback.")
    else:
        print("Done.")


if __name__ == "__main__":
    main()
