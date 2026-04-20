"""Day 9 — Messidor-2 paper figures.

Produces 300-dpi PNG + vector PDF in `results/figures/`:
  fig_g_messidor2_cm_sidebyside.{png,pdf}  — within-dataset Vanilla vs Ordinal CM
  fig_h_messidor2_roc.{png,pdf}            — cross-dataset binary ROC (4 models)
  fig_c_messidor2_uq_boxplot.{png,pdf}     — within-dataset UQ correct vs mis
  fig_d_messidor2_acc_coverage.{png,pdf}   — within-dataset 3-curve coverage

Usage:
    uv run python scripts/make_day9_figures.py [--strict]
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
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve

from src.uncertainty import digit_cap_entropy, prediction_margin


CLASS_NAMES = ["No DR", "Mild", "Moderate", "Severe", "PDR"]

WITHIN_VANILLA = Path("results/messidor2/capsnet_vanilla")
WITHIN_ORDINAL = Path("results/messidor2/ordinal_capsnet")
CROSS_ROOT = Path("results/cross_dataset/aptos_to_messidor2")

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


def _pool_within(dir_: Path, keys: tuple[str, ...]) -> dict:
    """Pool per-fold preds across 5 folds from a within-Messidor-2 model dir."""
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


# ---------------------------------------------------------------------------
# Fig G: within-Messidor-2 Vanilla vs Ordinal CM side-by-side
# ---------------------------------------------------------------------------

def _plot_cm(ax, cm_pct: np.ndarray, subtitle: str, cbar: bool) -> None:
    sns.heatmap(
        cm_pct, annot=True, fmt=".1f", cmap="Blues",
        xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=ax,
        cbar=cbar, vmin=0, vmax=100,
        annot_kws={"fontsize": 11},
        cbar_kws={"label": "% of true class"} if cbar else None,
    )
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title(subtitle, fontsize=12, pad=6)


def fig_g_sidebyside_cm() -> None:
    v = _pool_within(WITHIN_VANILLA, ("y_true", "y_pred"))
    o = _pool_within(WITHIN_ORDINAL, ("y_true", "y_pred"))
    cm_v = confusion_matrix(v["y_true"], v["y_pred"], labels=range(5))
    cm_o = confusion_matrix(o["y_true"], o["y_pred"], labels=range(5))
    denom_v = cm_v.sum(axis=1, keepdims=True).clip(min=1)
    denom_o = cm_o.sum(axis=1, keepdims=True).clip(min=1)
    cm_v_pct = cm_v.astype(float) / denom_v * 100
    cm_o_pct = cm_o.astype(float) / denom_o * 100

    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.2))
    _plot_cm(axes[0], cm_v_pct, "(a) Vanilla CapsNet  (Messidor-2 within)", cbar=False)
    _plot_cm(axes[1], cm_o_pct, "(b) Ordinal CapsNet  (Messidor-2 within)", cbar=True)
    fig.tight_layout()
    save_both(fig, "fig_g_messidor2_cm_sidebyside")
    print("  Figure G (Messidor-2 CM side-by-side) saved.")


# ---------------------------------------------------------------------------
# Fig H: cross-dataset binary ROC (4 models overlaid)
# ---------------------------------------------------------------------------

def _pool_cross_binary(model_subdir: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load the ensemble.npz for a cross-dataset model: (y_true_binary, binary_score)."""
    ens = np.load(model_subdir / "ensemble.npz")
    return ens["y_true_binary"], ens["binary_score"]


def fig_h_roc() -> None:
    curves = [
        ("MLP + CE",          CROSS_ROOT / "mlp_ce",          "#1f77b4"),
        ("MLP + MSE",         CROSS_ROOT / "mlp_mse",         "#2ca02c"),
        ("Vanilla CapsNet",   CROSS_ROOT / "capsnet_vanilla", "#ff7f0e"),
        ("Ordinal CapsNet",   CROSS_ROOT / "ordinal_capsnet", "#d62728"),
    ]
    fig, ax = plt.subplots(figsize=(6.5, 6.0))
    for name, d, color in curves:
        if not (d / "ensemble.npz").exists():
            print(f"  ROC: skipping {name} (no ensemble.npz)")
            continue
        y_true, score = _pool_cross_binary(d)
        fpr, tpr, _ = roc_curve(y_true, score)
        auc = roc_auc_score(y_true, score)
        ax.plot(fpr, tpr, color=color, linewidth=2.2,
                label=f"{name}   AUC = {auc:.3f}")
    ax.plot([0, 1], [0, 1], color="gray", linestyle="--", linewidth=1.2,
            label="Chance")
    ax.set_xlabel("False positive rate  (1 − specificity)")
    ax.set_ylabel("True positive rate  (sensitivity)")
    ax.set_xlim(-0.01, 1.01); ax.set_ylim(-0.01, 1.01)
    ax.grid(alpha=0.25, linestyle=":")
    ax.legend(loc="lower right", framealpha=0.95)
    fig.tight_layout()
    save_both(fig, "fig_h_messidor2_roc")
    print("  Figure H (Messidor-2 ROC) saved.")


# ---------------------------------------------------------------------------
# Fig C: within-Messidor-2 UQ boxplot
# ---------------------------------------------------------------------------

def fig_c_uq_boxplot() -> None:
    d = _pool_within(WITHIN_ORDINAL, ("y_true", "y_pred", "head_lengths"))
    hl = torch.from_numpy(d["head_lengths"])
    margin = prediction_margin(hl).numpy()
    correct = d["y_true"] == d["y_pred"]
    corr_v, mis_v = margin[correct], margin[~correct]
    if not len(corr_v) or not len(mis_v):
        print("  Figure C (Messidor-2) skipped: degenerate split")
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
    if p64 < 1e-300:        ptxt = "***   (p < 1e-300)"
    elif p64 < 0.001:       ptxt = f"***   (p = {p64:.2e})"
    elif p64 < 0.01:        ptxt = f"**    (p = {p64:.3f})"
    elif p64 < 0.05:        ptxt = f"*     (p = {p64:.3f})"
    else:                   ptxt = f"n.s.  (p = {p64:.3f})"
    ax.text(1.5, 1.095, f"Mann-Whitney U   {ptxt}",
            ha="center", va="bottom", fontsize=10)
    fig.tight_layout()
    save_both(fig, "fig_c_messidor2_uq_boxplot")
    print(f"  Figure C (Messidor-2 UQ boxplot) saved  |  p = {p64:.3e}")


# ---------------------------------------------------------------------------
# Fig D: within-Messidor-2 3-curve accuracy-coverage
# ---------------------------------------------------------------------------

def fig_d_acc_coverage() -> None:
    d = _pool_within(WITHIN_ORDINAL, ("y_true", "y_pred", "head_lengths", "routing_variance"))
    hl = torch.from_numpy(d["head_lengths"])
    pm = prediction_margin(hl).numpy()
    en = digit_cap_entropy(hl).numpy()
    correct = (d["y_true"] == d["y_pred"]).astype(int)
    baseline_acc = float(correct.mean())

    coverages = np.linspace(0.10, 1.0, 91)
    specs = [
        ("Prediction margin", pm, "#5c6bc0"),
        ("DigitCap entropy",  en, "#4c9f70"),
    ]
    if "routing_variance" in d:
        specs.append(("Routing variance", d["routing_variance"], "#c44e52"))
    else:
        print("  Figure D: routing_variance missing; restricted to 2 curves.")

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
    save_both(fig, "fig_d_messidor2_acc_coverage")
    print("  Figure D (Messidor-2 accuracy-coverage) saved.")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--strict", action="store_true")
    args = ap.parse_args()

    print(f"Writing Messidor-2 figures to {FIGDIR}/ ...")
    figs = [
        ("Figure G (Messidor-2 CM)",         fig_g_sidebyside_cm),
        ("Figure H (Messidor-2 ROC)",        fig_h_roc),
        ("Figure C (Messidor-2 UQ)",         fig_c_uq_boxplot),
        ("Figure D (Messidor-2 coverage)",   fig_d_acc_coverage),
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
