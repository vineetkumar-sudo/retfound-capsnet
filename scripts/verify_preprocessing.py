"""Verify preprocessing on APTOS images: original vs processed side by side."""

import sys
sys.path.insert(0, ".")

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.data.preprocessing import preprocess_fundus

# Grade 4 (proliferative DR) and Grade 0 (healthy)
samples = [
    ("data/aptos/train_images/001639a390f0.png", "Grade 4 (Proliferative DR)"),
    ("data/aptos/train_images/002c21358ce6.png", "Grade 0 (Healthy)"),
]

fig, axes = plt.subplots(len(samples), 2, figsize=(10, 5 * len(samples)))

for i, (img_path, label) in enumerate(samples):
    original = cv2.imread(img_path)
    original_rgb = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
    processed = preprocess_fundus(img_path)

    axes[i, 0].imshow(original_rgb)
    axes[i, 0].set_title(f"Original — {label}")
    axes[i, 0].axis("off")

    axes[i, 1].imshow(processed, cmap="gray")
    axes[i, 1].set_title(f"Processed — {label}")
    axes[i, 1].axis("off")

fig.tight_layout()
out_path = "results/preprocessing_sanity_check.png"
fig.savefig(out_path, dpi=150)
print(f"Saved comparison to {out_path}")
