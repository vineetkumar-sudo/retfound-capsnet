"""Day 7 — RETFound + Ordinal CapsNet architecture diagram.

Vector-exportable matplotlib figure showing the full pipeline: preprocessing ->
frozen RETFound backbone -> PrimaryCaps -> 4 parallel ordinal DigitCaps heads ->
grade sum + prediction-margin UQ side output. Outputs PDF (paper) + PNG (slides).

Usage:
    uv run python scripts/make_architecture_diagram.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


FIGDIR = Path("results/figures")
FIGDIR.mkdir(parents=True, exist_ok=True)

# Consistent styling shared with Day 6 figures
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 10,
})


# Palette
COL_INPUT = "#dce5f2"     # soft blue for data
COL_PREP = "#e8e4f7"      # lavender for preprocessing
COL_FROZEN = "#eeeeee"    # grey for frozen backbone
COL_CAPS = "#fff4cc"      # pale yellow for capsule layers
COL_HEAD = "#ffe0cc"      # peach for heads
COL_OUT = "#d8f3d8"       # light green for outputs
COL_UQ = "#f9d5d5"        # light coral for UQ side output

EDGE = "#333333"
TEXT = "#111111"


def box(ax, xy, w, h, label, facecolor, dashed=False, fontsize=10, weight="normal"):
    """Rounded rectangle with centred label."""
    style = "round,pad=0.02,rounding_size=0.08"
    patch = FancyBboxPatch(
        xy, w, h, boxstyle=style,
        facecolor=facecolor, edgecolor=EDGE,
        linewidth=1.3, linestyle=("--" if dashed else "-"),
    )
    ax.add_patch(patch)
    cx, cy = xy[0] + w / 2, xy[1] + h / 2
    ax.text(cx, cy, label, ha="center", va="center",
            fontsize=fontsize, color=TEXT, wrap=True, weight=weight)


def arrow(ax, start, end, text=None, text_offset=(0.05, 0.0), color=EDGE,
          dashed=False, curve="arc3,rad=0.0", lw=1.3):
    a = FancyArrowPatch(
        start, end, arrowstyle="-|>", mutation_scale=14,
        color=color, linewidth=lw, connectionstyle=curve,
        linestyle=("--" if dashed else "-"),
    )
    ax.add_patch(a)
    if text:
        mx = (start[0] + end[0]) / 2 + text_offset[0]
        my = (start[1] + end[1]) / 2 + text_offset[1]
        ax.text(mx, my, text, fontsize=9, color=TEXT, ha="left", va="center")


def main() -> None:
    # Wider canvas: 4 heads need full span across x, and the UQ box lives in the
    # bottom-right corner so it doesn't collide with the rightmost head.
    fig, ax = plt.subplots(figsize=(13.5, 10.0))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 11)
    ax.set_axis_off()
    ax.set_aspect("equal")

    cx = 7.0  # horizontal centreline for the vertical trunk

    # ---- Input ----
    box(ax, (cx - 1.6, 10.0), 3.2, 0.75,
        "Fundus image  (H×W×3)",
        COL_INPUT, fontsize=11, weight="bold")

    # ---- Preprocessing ----
    box(ax, (cx - 2.4, 8.6), 4.8, 0.95,
        "Resize 256 (bicubic)  ·  CenterCrop 224  ·  ImageNet Normalize\n(matches official RETFound eval transform)",
        COL_PREP, fontsize=9)
    arrow(ax, (cx, 10.0), (cx, 9.55))

    # ---- Frozen backbone (dashed border) ----
    box(ax, (cx - 2.6, 6.95), 5.2, 1.25,
        "RETFound ViT-L  (frozen, nature-CFP MAE)\nforward pass → CLS token ∈ ℝ¹⁰²⁴",
        COL_FROZEN, dashed=True, fontsize=10, weight="bold")
    arrow(ax, (cx, 8.6), (cx, 8.22))
    ax.text(cx + 2.8, 7.6, "no gradients\nfrozen weights",
            fontsize=8, color="#555555", style="italic")

    # ---- PrimaryCaps ----
    box(ax, (cx - 2.2, 5.35), 4.4, 1.05,
        "PrimaryCaps   Linear 1024 → 256\nreshape → 32 capsules × 8  ·  Sabour squash",
        COL_CAPS, fontsize=10)
    arrow(ax, (cx, 6.95), (cx, 6.40))
    ax.text(cx + 2.4, 6.68, "CLS  (B, 1024)", fontsize=9, color=TEXT)

    # ---- Fan-out to 4 heads ----
    trunk_y = 5.0
    arrow(ax, (cx, 5.35), (cx, trunk_y))
    ax.text(cx + 2.4, 5.1, "primary caps  (B, 32, 8)", fontsize=9, color=TEXT)

    # 4 heads evenly spread across the full x-range
    head_w = 2.6
    head_h = 1.4
    head_y = 3.2
    head_specs = [
        ("Head 0", "P(y > 0)"),
        ("Head 1", "P(y > 1)"),
        ("Head 2", "P(y > 2)"),
        ("Head 3", "P(y > 3)"),
    ]
    # Distribute 4 boxes across x=0.5 .. 13.5 leaving uniform gaps
    n_heads = 4
    gap = (14.0 - 0.5 * 2 - head_w * n_heads) / (n_heads - 1)
    head_xs = [0.5 + i * (head_w + gap) for i in range(n_heads)]
    for hx, (hname, hlabel) in zip(head_xs, head_specs):
        box(ax, (hx, head_y), head_w, head_h,
            f"{hname}  —  DigitCaps\n2 caps × 16  ·  3-iter dynamic routing\n{hlabel}",
            COL_HEAD, fontsize=9)
        hx_center = hx + head_w / 2
        arrow(ax, (cx, trunk_y), (hx_center, head_y + head_h),
              curve="arc3,rad=0.0", lw=1.1)

    # ---- Pooled per-head outputs layer ----
    pool_x, pool_w = 2.0, 10.0
    box(ax, (pool_x, 1.90), pool_w, 0.75,
        "per-head positive-capsule length  ∈ [0, 1]⁴      →      threshold @ 0.5",
        COL_CAPS, fontsize=9)
    for hx in head_xs:
        hx_center = hx + head_w / 2
        arrow(ax, (hx_center, head_y), (hx_center, 2.65),
              curve="arc3,rad=0.0", lw=1.0)

    # ---- Grade inference (bottom-left) ----
    box(ax, (1.8, 0.55), 6.2, 0.95,
        "Predicted grade  ŷ = Σₖ 𝟙[P(y > k) > 0.5]   ∈ {0,1,2,3,4}\n(monotonic decoding → fewer distant errors)",
        COL_OUT, fontsize=10, weight="bold")
    arrow(ax, (4.9, 1.90), (4.9, 1.50))

    # ---- UQ side output (bottom-right) ----
    uq_x, uq_y, uq_w, uq_h = 9.0, 0.55, 4.0, 0.95
    box(ax, (uq_x, uq_y), uq_w, uq_h,
        "UQ: prediction margin = 1 − (top1 − top2)\nover 5-class probs (chain-rule from heads)",
        COL_UQ, fontsize=9)
    # Dashed tap arrow from the pooled layer into the UQ box
    arrow(ax, (11.0, 1.90), (uq_x + uq_w / 2, uq_y + uq_h),
          dashed=True, lw=1.1, curve="arc3,rad=-0.15")
    ax.text(11.15, 1.65, "(side output, no gradient into routing)",
            fontsize=8, color="#555555", style="italic")

    # ---- Legend (frozen / trainable) ----
    legend_x, legend_y = 0.3, 10.25
    ax.add_patch(FancyBboxPatch((legend_x, legend_y), 0.4, 0.25,
                                boxstyle="round,pad=0.01,rounding_size=0.04",
                                facecolor=COL_FROZEN, edgecolor=EDGE,
                                linewidth=1.1, linestyle="--"))
    ax.text(legend_x + 0.5, legend_y + 0.125, "frozen (no gradient)",
            fontsize=8.5, va="center")
    ax.add_patch(FancyBboxPatch((legend_x, legend_y - 0.45), 0.4, 0.25,
                                boxstyle="round,pad=0.01,rounding_size=0.04",
                                facecolor=COL_CAPS, edgecolor=EDGE, linewidth=1.1))
    ax.text(legend_x + 0.5, legend_y + 0.125 - 0.45,
            "trainable capsule layers",
            fontsize=8.5, va="center")

    fig.tight_layout()
    fig.savefig(FIGDIR / "fig_architecture.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIGDIR / "fig_architecture.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote:\n  {FIGDIR/'fig_architecture.png'}\n  {FIGDIR/'fig_architecture.pdf'}")


if __name__ == "__main__":
    main()
