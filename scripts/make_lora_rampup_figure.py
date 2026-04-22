"""LoRA convergence ramp-up figure (bonus Fig E').

Plots per-epoch val QWK across all 15 folds (3 seeds × 5 folds) from the
LoRA overnight run, with the per-fold curves as thin lines plus a bold
mean ± std band, horizontal reference lines at the frozen Ordinal
baseline (0.8932) and LoRA pooled 3-seed mean (0.9127), and tick markers
for the best-epoch distribution across folds.

Reads `results/lora_ordinal_capsnet/seed{S}/summary.json::per_fold[*].history`
(written by `scripts/run_lora_ordinal.py`).

Usage:
    uv run python scripts/make_lora_rampup_figure.py
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


LORA_DIR = Path("results/lora_ordinal_capsnet")
FIGDIR = Path("results/figures")
FIGDIR.mkdir(parents=True, exist_ok=True)

FROZEN_QWK = 0.8932  # Ordinal CapsNet 3-seed mean (row 6 in aptos_final_table.md)
LORA_QWK = 0.9127    # Ordinal CapsNet + LoRA 3-seed mean (row 9)
MAX_EPOCHS = 20      # training cap — from cfg

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 12,
    "axes.labelsize": 13,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 10,
    "savefig.bbox": "tight",
})


def load_all_folds() -> tuple[np.ndarray, list[int]]:
    """Return (val_qwk_matrix, best_epochs) pooled across every seed/fold.

    val_qwk_matrix: (15, MAX_EPOCHS) — NaN where a fold early-stopped before
    reaching that epoch.
    best_epochs: list of int (1-indexed) of best-QWK epoch per fold.
    """
    rows: list[np.ndarray] = []
    best_epochs: list[int] = []
    for sd in sorted(LORA_DIR.glob("seed*")):
        summary = json.loads((sd / "summary.json").read_text())
        for fold in summary["per_fold"]:
            hist = fold.get("history", [])
            q = np.full(MAX_EPOCHS, np.nan, dtype=np.float64)
            for entry in hist:
                ep = int(entry["epoch"])
                if 1 <= ep <= MAX_EPOCHS:
                    q[ep - 1] = float(entry["val_qwk"])
            rows.append(q)
            best_epochs.append(int(fold["best_epoch"]))
    if not rows:
        raise FileNotFoundError(f"No seed*/summary.json found under {LORA_DIR}")
    return np.stack(rows, axis=0), best_epochs


def main() -> None:
    matrix, best_epochs = load_all_folds()
    n_folds = matrix.shape[0]
    epochs = np.arange(1, MAX_EPOCHS + 1)

    # Mean / std ignoring NaNs (early-stopped epochs)
    with np.errstate(invalid="ignore"):
        mean = np.nanmean(matrix, axis=0)
        std = np.nanstd(matrix, axis=0)

    # Median best-epoch for the annotation — more robust than mean across a bimodal set
    best_median = int(np.median(best_epochs))
    best_min = int(np.min(best_epochs))
    best_max = int(np.max(best_epochs))

    fig, ax = plt.subplots(figsize=(7.5, 5.0))

    # Individual fold curves (thin, translucent)
    for r in range(n_folds):
        ax.plot(epochs, matrix[r], color="#5c6bc0", alpha=0.22, linewidth=1.0)

    # Mean ± std band + bold mean
    ax.fill_between(epochs, mean - std, mean + std,
                    color="#5c6bc0", alpha=0.20, label=f"±1 SD across {n_folds} folds")
    ax.plot(epochs, mean, color="#1a237e", linewidth=2.6,
            label=f"Mean across {n_folds} folds (3 seeds × 5 folds)")

    # Reference lines — frozen baseline + LoRA final
    ax.axhline(FROZEN_QWK, linestyle="--", color="#888", linewidth=1.4,
               label=f"Frozen Ordinal CapsNet (0.8932)")
    ax.axhline(LORA_QWK, linestyle=":", color="#c44e52", linewidth=1.6,
               label=f"LoRA pooled mean (0.9127)")

    # Best-epoch distribution markers along the bottom
    y0 = ax.get_ylim()[0] if False else 0.42  # fixed floor for clean visual
    ax.set_ylim(0.42, 0.975)
    best_y = 0.455
    for ep in best_epochs:
        ax.plot(ep, best_y, marker="v", markersize=7,
                markerfacecolor="#c44e52", markeredgecolor="black",
                markeredgewidth=0.4)
    # Annotate best-epoch statistic
    ax.annotate(
        f"Best epoch per fold\n"
        f"median {best_median}, range [{best_min}, {best_max}]",
        xy=(best_median, best_y),
        xytext=(best_median + 3, best_y + 0.05),
        fontsize=10, ha="left",
        arrowprops=dict(arrowstyle="->", color="#c44e52", lw=1.0, alpha=0.8),
    )

    # "Converges above frozen baseline at epoch N" callout
    for ep in epochs:
        if not np.isnan(mean[ep - 1]) and mean[ep - 1] > FROZEN_QWK:
            ax.axvline(ep, linestyle=":", color="#4c9f70", alpha=0.5, linewidth=1.0)
            ax.annotate(
                f"Mean > frozen baseline\nat epoch {ep}",
                xy=(ep, FROZEN_QWK),
                xytext=(ep + 1.4, FROZEN_QWK - 0.08),
                fontsize=10, color="#2e5e3e",
                arrowprops=dict(arrowstyle="->", color="#4c9f70", lw=0.9, alpha=0.8),
            )
            break

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Val QWK")
    ax.set_xlim(0.5, MAX_EPOCHS + 0.5)
    ax.set_xticks([1, 5, 10, 15, 20])
    ax.grid(alpha=0.25, linestyle=":")
    ax.legend(loc="lower right", framealpha=0.95)
    fig.tight_layout()

    out_stem = "fig_e_lora_rampup"
    fig.savefig(FIGDIR / f"{out_stem}.png", dpi=300)
    fig.savefig(FIGDIR / f"{out_stem}.pdf")
    plt.close(fig)

    print(f"Wrote  {FIGDIR}/{out_stem}.png  +  .pdf")
    print(f"  n_folds aggregated:    {n_folds}")
    print(f"  best-epoch median:     {best_median}  (range [{best_min}, {best_max}])")
    print(f"  mean QWK @ ep 1:       {mean[0]:.4f}")
    print(f"  mean QWK @ ep 5:       {mean[4]:.4f}")
    print(f"  mean QWK @ ep 9:       {mean[8]:.4f}")
    print(f"  mean QWK @ ep 15:      {mean[14]:.4f}")


if __name__ == "__main__":
    main()
