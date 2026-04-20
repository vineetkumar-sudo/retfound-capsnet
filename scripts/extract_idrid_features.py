"""Day 8 — Extract RETFound CLS features for the IDRiD B. Disease Grading split.

Mirrors the APTOS feature-cache pipeline (src/data/feature_cache.py) but handles:
  - IDRiD's directory layout ('B. Disease Grading/1. Original Images/...').
  - .jpg image extension (APTOS uses .png).
  - CSV columns 'Image name' and 'Retinopathy grade'.
  - Both train AND test splits have labels (APTOS test is unlabeled).

Outputs (idempotent — wipes output dir first):
    data/idrid/features/train_features.npy  (413, 1024) float32
    data/idrid/features/train_labels.npy    (413,)      int64
    data/idrid/features/train_ids.npy       (413,)      str
    data/idrid/features/test_features.npy   (103, 1024) float32
    data/idrid/features/test_labels.npy     (103,)      int64
    data/idrid/features/test_ids.npy        (103,)      str

Usage:
    uv run python scripts/extract_idrid_features.py
"""

from __future__ import annotations

import csv
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, ".")

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from src.data.feature_cache import RETFOUND_TRANSFORM, load_retfound


IDRID_ROOT = Path("data/idrid/B. Disease Grading")
IMG_ROOT = IDRID_ROOT / "1. Original Images"
LABEL_ROOT = IDRID_ROOT / "2. Groundtruths"
OUT_DIR = Path("data/idrid/features")
WEIGHTS = Path("data/weights/RETFound_mae_natureCFP.pth")


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_csv_labels(csv_path: Path) -> tuple[list[str], list[int]]:
    """Parse IDRiD labels CSV (may have trailing empty columns)."""
    ids, labels = [], []
    with csv_path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            img_id = (row.get("Image name") or "").strip()
            if not img_id:
                continue
            grade_raw = (row.get("Retinopathy grade") or "").strip()
            if grade_raw == "":
                continue
            ids.append(img_id)
            labels.append(int(grade_raw))
    return ids, labels


@torch.no_grad()
def extract_features(
    model: torch.nn.Module,
    image_dir: Path,
    image_ids: list[str],
    device: torch.device,
    batch_size: int,
    split_name: str,
) -> np.ndarray:
    """Run RETFound forward pass on IDRiD .jpg images, collect CLS tokens."""
    out_chunks: list[np.ndarray] = []
    t0 = time.time()
    for i in tqdm(
        range(0, len(image_ids), batch_size),
        desc=f"  {split_name}", unit="batch",
    ):
        batch_ids = image_ids[i:i + batch_size]
        tensors = []
        for img_id in batch_ids:
            img_path = image_dir / f"{img_id}.jpg"
            img = Image.open(img_path).convert("RGB")
            tensors.append(RETFOUND_TRANSFORM(img))
        batch = torch.stack(tensors).to(device)
        feats = model.forward_features(batch)  # (B, 197, 1024)
        cls = feats[:, 0]                      # (B, 1024)
        out_chunks.append(cls.cpu().numpy())
    elapsed = time.time() - t0
    print(f"    Done: {len(image_ids)} images in {elapsed:.1f}s "
          f"({len(image_ids) / elapsed:.1f} img/s)")
    return np.concatenate(out_chunks, axis=0)


def main() -> None:
    assert IDRID_ROOT.exists(), f"Missing {IDRID_ROOT}. Unzip IDRiD under data/idrid/ first."
    assert WEIGHTS.exists(), f"Missing RETFound weights at {WEIGHTS}"

    # Idempotent: wipe any previous cache
    if OUT_DIR.exists():
        print(f"Clearing previous cache at {OUT_DIR}/")
        shutil.rmtree(OUT_DIR)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    device = get_device()
    print(f"Device: {device}")
    print("Loading RETFound weights...")
    t0 = time.time()
    model = load_retfound(str(WEIGHTS), device)
    print(f"Model loaded in {time.time() - t0:.1f}s\n")

    for split_name, img_subdir, csv_name in [
        ("Train", "a. Training Set",
         "a. IDRiD_Disease Grading_Training Labels.csv"),
        ("Test", "b. Testing Set",
         "b. IDRiD_Disease Grading_Testing Labels.csv"),
    ]:
        print(f"=== {split_name} split ===")
        csv_path = LABEL_ROOT / csv_name
        img_dir = IMG_ROOT / img_subdir
        ids, labels = load_csv_labels(csv_path)
        print(f"  {csv_path.name}: {len(ids)} labelled rows")
        print(f"  Label distribution (grades 0..4): {np.bincount(np.array(labels), minlength=5).tolist()}")

        feats = extract_features(model, img_dir, ids, device, batch_size=32,
                                 split_name=split_name)
        labels_arr = np.asarray(labels, dtype=np.int64)
        assert not np.isnan(feats).any(), "NaNs in extracted features"
        print(f"  features shape: {feats.shape}, dtype: {feats.dtype}")

        prefix = split_name.lower()
        np.save(OUT_DIR / f"{prefix}_features.npy", feats)
        np.save(OUT_DIR / f"{prefix}_labels.npy", labels_arr)
        np.save(OUT_DIR / f"{prefix}_ids.npy", np.array(ids))
        print(f"  Saved -> {OUT_DIR}/{prefix}_{{features,labels,ids}}.npy\n")

    total_mb = sum(f.stat().st_size for f in OUT_DIR.glob("*.npy")) / 1024 / 1024
    print("=" * 40)
    print(f"IDRiD feature cache: {total_mb:.1f} MB total")
    for f in sorted(OUT_DIR.glob("*.npy")):
        arr = np.load(f, mmap_mode="r")
        print(f"  {f.name:25s} shape={str(arr.shape):15s} dtype={arr.dtype}")


if __name__ == "__main__":
    main()
