"""Day 9 — Extract RETFound CLS features for Messidor-2.

Pipeline mirrors APTOS + IDRiD but handles Messidor-2's specifics:
  - Label source: messidor_data.csv (Google Brain adjudicated grades on Kaggle;
    1,748 rows, columns: image_id, adjudicated_dr_grade, adjudicated_dme,
    adjudicated_gradable). 4 rows have adjudicated_gradable=0 and are dropped.
  - Filenames: CSV records every `image_id` with a `.png` extension but the
    images ADCIS ships are a mix of `.png` (Topcon TRC NW6) and `.JPG` (Canon
    CR-DGi 8.5 MP). We stem-match to the actual file on disk.
  - Binary label for referable DR: 1 if adjudicated_dr_grade in {2,3,4}, else 0.

Outputs (idempotent, wipes output dir first):
    data/messidor2/features/features.npy         (N, 1024)  float32
    data/messidor2/features/grades.npy           (N,)       int64  (0-4)
    data/messidor2/features/binary_labels.npy    (N,)       int64  (0 or 1)
    data/messidor2/features/image_ids.npy        (N,)       str
    data/messidor2/features/dme.npy              (N,)       int64

Fails loud if:
    - messidor_data.csv or images dir missing
    - any gradable row has no matching image on disk
    - features contain NaNs

Usage:
    uv run python scripts/extract_messidor2_features.py
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
from src.utils import enable_tf32


ROOT = Path("data/messidor2")
IMG_DIR = ROOT / "IMAGES"
LABELS_CSV = ROOT / "messidor_data.csv"
OUT_DIR = ROOT / "features"
WEIGHTS = Path("data/weights/RETFound_mae_natureCFP.pth")


def get_device() -> torch.device:
    if torch.cuda.is_available():
        enable_tf32()
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_filename_index(img_dir: Path) -> dict[str, Path]:
    """Map filename stem -> actual file path, covering .png/.jpg/.JPG variants."""
    index: dict[str, Path] = {}
    for ext in ("*.png", "*.jpg", "*.JPG"):
        for p in img_dir.glob(ext):
            index[p.stem] = p
    return index


def load_labels(csv_path: Path) -> list[dict]:
    """Parse the adjudicated labels; skip rows with adjudicated_gradable != 1."""
    rows: list[dict] = []
    with csv_path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            if str(row.get("adjudicated_gradable", "")).strip() != "1":
                continue
            grade_raw = str(row.get("adjudicated_dr_grade", "")).strip()
            dme_raw = str(row.get("adjudicated_dme", "")).strip()
            if grade_raw == "" or grade_raw.lower() == "nan":
                continue
            rows.append({
                "image_id": row["image_id"].strip(),
                "grade": int(grade_raw),
                "dme": int(dme_raw) if dme_raw.isdigit() else -1,
            })
    return rows


@torch.no_grad()
def extract_features(
    model: torch.nn.Module, items: list[dict], name_index: dict[str, Path],
    device: torch.device, batch_size: int = 32,
) -> np.ndarray:
    """Forward each item's image through RETFound; return (N, 1024) CLS features."""
    chunks = []
    t0 = time.time()
    for i in tqdm(range(0, len(items), batch_size), desc="  Messidor-2", unit="batch"):
        batch = items[i:i + batch_size]
        tensors = []
        for it in batch:
            stem = Path(it["image_id"]).stem
            path = name_index[stem]   # KeyError here would be a missing image — caller already verified
            img = Image.open(path).convert("RGB")
            tensors.append(RETFOUND_TRANSFORM(img))
        x = torch.stack(tensors).to(device)
        feats = model.forward_features(x)[:, 0]   # (B, 1024)
        chunks.append(feats.cpu().numpy())
    elapsed = time.time() - t0
    print(f"    Done: {len(items)} images in {elapsed:.1f}s ({len(items)/elapsed:.1f} img/s)")
    return np.concatenate(chunks, axis=0)


def main() -> None:
    assert ROOT.exists(), f"Missing {ROOT}/"
    assert IMG_DIR.exists(), f"Missing {IMG_DIR}/"
    assert LABELS_CSV.exists(), f"Missing {LABELS_CSV} — run kaggle download first"
    assert WEIGHTS.exists(), f"Missing RETFound weights at {WEIGHTS}"

    if OUT_DIR.exists():
        print(f"Clearing previous cache at {OUT_DIR}/")
        shutil.rmtree(OUT_DIR)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # --- Labels ---
    items = load_labels(LABELS_CSV)
    print(f"Labels CSV: {LABELS_CSV}")
    print(f"  {len(items)} gradable rows")
    grades = np.array([it["grade"] for it in items], dtype=np.int64)
    binary = (grades >= 2).astype(np.int64)
    print(f"  grade distribution (0..4): {np.bincount(grades, minlength=5).tolist()}")
    print(f"  binary (referable=1): {np.bincount(binary).tolist()}  "
          f"({binary.mean() * 100:.1f}% referable)")

    # --- Filename join (fail loud on any miss) ---
    name_index = build_filename_index(IMG_DIR)
    print(f"\nImage dir stems indexed: {len(name_index)}")
    missing = [it["image_id"] for it in items if Path(it["image_id"]).stem not in name_index]
    if missing:
        raise FileNotFoundError(
            f"{len(missing)} labels have no matching image on disk. "
            f"First 5: {missing[:5]}"
        )
    print(f"  Every label joined to an image (stem-match).")

    # --- RETFound ---
    device = get_device()
    print(f"\nDevice: {device}")
    print("Loading RETFound...")
    t0 = time.time()
    model = load_retfound(str(WEIGHTS), device)
    print(f"Model loaded in {time.time() - t0:.1f}s\n")

    feats = extract_features(model, items, name_index, device)
    assert not np.isnan(feats).any(), "NaNs in extracted features"
    assert feats.shape == (len(items), 1024), f"Unexpected shape: {feats.shape}"

    # --- Save ---
    ids = np.array([it["image_id"] for it in items])
    dme = np.array([it["dme"] for it in items], dtype=np.int64)
    np.save(OUT_DIR / "features.npy", feats)
    np.save(OUT_DIR / "grades.npy", grades)
    np.save(OUT_DIR / "binary_labels.npy", binary)
    np.save(OUT_DIR / "image_ids.npy", ids)
    np.save(OUT_DIR / "dme.npy", dme)

    total_mb = sum(f.stat().st_size for f in OUT_DIR.glob("*.npy")) / 1024 / 1024
    print(f"\nMessidor-2 feature cache: {total_mb:.1f} MB")
    for f in sorted(OUT_DIR.glob("*.npy")):
        arr = np.load(f, mmap_mode="r")
        print(f"  {f.name:20s} shape={str(arr.shape):15s} dtype={arr.dtype}")


if __name__ == "__main__":
    main()
