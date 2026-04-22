"""Tiled 1x5 figure of representative APTOS samples — one per ICDR grade (0-4).

Intended for the paper's Introduction as a "what the DR scale looks like"
anchor. Picks one deterministic sample per grade from APTOS train using
the same preprocessing pipeline as our models (Resize 256 bicubic ->
CenterCrop 224), then tiles them with grade labels + short clinical
descriptions.

This script **does not touch any experimental result**. It only reads
raw APTOS images from `data/aptos/train_images/` and writes a single
figure to `results/figures/`.

Usage:
    uv run python scripts/make_grade_samples_figure.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, ".")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from torchvision import transforms as T

APTOS_CSV = Path("data/aptos/train.csv")
APTOS_IMG_DIR = Path("data/aptos/train_images")
OUT_DIR = Path("results/figures")
OUT_STEM = "fig_0_grade_samples"

# Short ICDR-severity-scale descriptions — drop into the paper caption verbatim.
GRADE_INFO = [
    ("0", "No DR",             "No visible microvascular lesions"),
    ("1", "Mild NPDR",         "Microaneurysms only"),
    ("2", "Moderate NPDR",     "MAs + haemorrhages, hard exudates, or cotton-wool spots"),
    ("3", "Severe NPDR",       "Extensive haemorrhages / IRMA / venous beading"),
    ("4", "Proliferative DR",  "Neovascularisation or vitreous/pre-retinal haemorrhage"),
]

# Seed used here ONLY to pick a prototype image per grade — does not touch
# any model RNG. Training/eval splits elsewhere still use cfg.data.seed=42.
PICK_SEED = 7

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.titlesize": 12,
    "savefig.bbox": "tight",
})


def load_labels() -> list[tuple[str, int]]:
    """Return list of (id_code, grade)."""
    import csv
    out: list[tuple[str, int]] = []
    with APTOS_CSV.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            out.append((row["id_code"], int(row["diagnosis"])))
    return out


def pick_one_per_grade(entries: list[tuple[str, int]], seed: int = PICK_SEED) -> dict[int, str]:
    """Deterministically pick one id_code per grade using a fixed RNG."""
    by_grade: dict[int, list[str]] = {g: [] for g in range(5)}
    for id_code, g in entries:
        by_grade[g].append(id_code)
    rng = np.random.default_rng(seed)
    chosen: dict[int, str] = {}
    for g in range(5):
        lst = sorted(by_grade[g])  # stable
        idx = int(rng.integers(0, len(lst)))
        chosen[g] = lst[idx]
    return chosen


def load_and_preprocess(id_code: str) -> np.ndarray:
    """Return a 224x224x3 uint8 array after the official RETFound eval transform."""
    path = APTOS_IMG_DIR / f"{id_code}.png"
    img = Image.open(path).convert("RGB")
    # Match the preprocessing in src/data/feature_cache.py (Resize 256 bicubic
    # + CenterCrop 224), but without the ImageNet normalise — we want the
    # human-visible pixels for the figure, not the normalised tensor.
    tr = T.Compose([
        T.Resize(256, interpolation=T.InterpolationMode.BICUBIC),
        T.CenterCrop(224),
    ])
    img = tr(img)
    return np.asarray(img)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    entries = load_labels()
    chosen = pick_one_per_grade(entries)

    fig, axes = plt.subplots(1, 5, figsize=(15.0, 3.6))
    for ax, (grade_str, name, desc) in zip(axes, GRADE_INFO):
        g = int(grade_str)
        id_code = chosen[g]
        img = load_and_preprocess(id_code)
        ax.imshow(img)
        ax.set_xticks([]); ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_title(f"Grade {grade_str} — {name}", fontweight="bold", pad=6)
        # Short clinical descriptor under each panel
        ax.text(0.5, -0.08, desc,
                transform=ax.transAxes,
                ha="center", va="top",
                fontsize=9, color="black", wrap=True)
        # Sample id in small type at the bottom right for traceability
        ax.text(0.99, 0.02, f"{id_code}",
                transform=ax.transAxes,
                ha="right", va="bottom",
                fontsize=7, color="white",
                bbox=dict(boxstyle="round,pad=0.12",
                          facecolor="black", alpha=0.35,
                          edgecolor="none"))
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.20)
    fig.savefig(OUT_DIR / f"{OUT_STEM}.png", dpi=300)
    fig.savefig(OUT_DIR / f"{OUT_STEM}.pdf")
    plt.close(fig)

    print(f"Wrote  {OUT_DIR}/{OUT_STEM}.png  +  .pdf")
    print("Selected APTOS samples (deterministic, PICK_SEED=" + str(PICK_SEED) + "):")
    for g in range(5):
        print(f"  Grade {g}: {chosen[g]}")


if __name__ == "__main__":
    main()
