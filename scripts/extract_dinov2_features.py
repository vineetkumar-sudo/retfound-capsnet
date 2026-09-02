"""Day-12 backbone-comparison ablation: extract DINOv2 ViT-L/14 CLS features
for APTOS-2019 train and Messidor-2 (gradable).

Why: the paper's current single-backbone story (RETFound only) is a reviewer
red-flag — recent head-to-head benchmarks show DINOv2 sometimes outperforms
RETFound on frozen DR downstream tasks. We run the same Ordinal CapsNet and
MLP + K-1 sigmoid heads on DINOv2 CLS features to disentangle *head value*
from *backbone choice*.

Design:
  - ViT-L/14 via torch.hub facebookresearch/dinov2. CLS output is 1024-dim,
    matching RETFound's feature dim, so downstream configs stay identical.
  - Preprocessing = our standard eval transform (Resize-256 bicubic →
    CenterCrop-224 → ImageNet normalise). DINOv2's official eval transform
    uses the same pipeline, so the only variable changing between RETFound
    and DINOv2 runs is the backbone weights.
  - Single model instance loaded once; we extract APTOS and Messidor-2 back
    to back to amortise the ~1.2 GB weight download and the model-init cost.

Outputs (idempotent; wipes each output dir first):
  data/aptos/features_dinov2/{train_features,train_labels,train_ids}.npy
  data/messidor2/features_dinov2/{features,grades,binary_labels,image_ids,dme}.npy

Usage:
  uv run python scripts/extract_dinov2_features.py
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
from torchvision import transforms
from tqdm import tqdm
from src.utils import enable_tf32

# Mirror src/data/feature_cache.py::RETFOUND_TRANSFORM exactly, so the only
# variable is the backbone (DINOv2 official eval transform is the same).
EVAL_TRANSFORM = transforms.Compose([
    transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


def get_device() -> torch.device:
    if torch.cuda.is_available():
        enable_tf32()
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_dinov2(device: torch.device) -> torch.nn.Module:
    """Load frozen DINOv2 ViT-L/14 from the official torch.hub repo.

    First run downloads ~1.2 GB into ~/.cache/torch/hub/. Subsequent runs are
    free. We freeze every parameter and switch to eval mode; forward(x)
    returns the CLS token feature (B, 1024).
    """
    print("Loading DINOv2 ViT-L/14 from torch.hub (facebookresearch/dinov2)...")
    t0 = time.time()
    model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitl14")
    model.eval()
    model.to(device)
    for p in model.parameters():
        p.requires_grad = False
    print(f"  loaded in {time.time() - t0:.1f}s")
    return model


@torch.no_grad()
def extract_batch_features(
    model: torch.nn.Module,
    image_paths: list[Path],
    device: torch.device,
    batch_size: int,
    label: str,
) -> np.ndarray:
    """Generic CLS extraction; returns (N, 1024) float32."""
    chunks: list[np.ndarray] = []
    t0 = time.time()
    total_batches = (len(image_paths) + batch_size - 1) // batch_size
    pbar = tqdm(range(0, len(image_paths), batch_size), desc=f"  {label}", unit="batch", total=total_batches)
    for i in pbar:
        batch = image_paths[i : i + batch_size]
        tensors = [EVAL_TRANSFORM(Image.open(p).convert("RGB")) for p in batch]
        x = torch.stack(tensors).to(device)
        cls = model(x)            # (B, 1024) — DINOv2 hub model returns CLS directly
        chunks.append(cls.cpu().numpy().astype(np.float32))
    arr = np.concatenate(chunks, axis=0)
    elapsed = time.time() - t0
    print(f"    Done: {len(image_paths)} images in {elapsed:.1f}s ({len(image_paths)/elapsed:.1f} img/s)")
    return arr


# ---------------------------------------------------------------------------
# APTOS
# ---------------------------------------------------------------------------

APTOS_ROOT = Path("data/aptos")
APTOS_OUT = APTOS_ROOT / "features_dinov2"


def extract_aptos(model: torch.nn.Module, device: torch.device, batch_size: int) -> None:
    print("\n=== APTOS-2019 train split ===")
    if APTOS_OUT.exists():
        print(f"Clearing previous cache at {APTOS_OUT}/")
        shutil.rmtree(APTOS_OUT)
    APTOS_OUT.mkdir(parents=True, exist_ok=True)

    ids, labels = [], []
    with (APTOS_ROOT / "train.csv").open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            ids.append(row["id_code"])
            labels.append(int(row["diagnosis"]))
    paths = [APTOS_ROOT / "train_images" / f"{i}.png" for i in ids]
    print(f"  {len(paths)} images")

    feats = extract_batch_features(model, paths, device, batch_size, "APTOS")
    assert not np.isnan(feats).any(), "NaNs in APTOS DINOv2 features"
    assert feats.shape == (len(paths), 1024), f"shape {feats.shape} unexpected"

    np.save(APTOS_OUT / "train_features.npy", feats)
    np.save(APTOS_OUT / "train_labels.npy", np.array(labels, dtype=np.int64))
    np.save(APTOS_OUT / "train_ids.npy", np.array(ids))
    print(f"  Saved -> {APTOS_OUT}/")


# ---------------------------------------------------------------------------
# Messidor-2 (mirrors extract_messidor2_features.py's label join)
# ---------------------------------------------------------------------------

MESS_ROOT = Path("data/messidor2")
MESS_IMG = MESS_ROOT / "IMAGES"
MESS_CSV = MESS_ROOT / "messidor_data.csv"
MESS_OUT = MESS_ROOT / "features_dinov2"


def build_filename_index(img_dir: Path) -> dict[str, Path]:
    index: dict[str, Path] = {}
    for ext in ("*.png", "*.jpg", "*.JPG"):
        for p in img_dir.glob(ext):
            index[p.stem] = p
    return index


def load_messidor2_labels(csv_path: Path) -> list[dict]:
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


def extract_messidor2(model: torch.nn.Module, device: torch.device, batch_size: int) -> None:
    print("\n=== Messidor-2 (gradable) ===")
    assert MESS_ROOT.exists() and MESS_IMG.exists() and MESS_CSV.exists(), \
        f"Messidor-2 layout missing under {MESS_ROOT}"

    if MESS_OUT.exists():
        print(f"Clearing previous cache at {MESS_OUT}/")
        shutil.rmtree(MESS_OUT)
    MESS_OUT.mkdir(parents=True, exist_ok=True)

    items = load_messidor2_labels(MESS_CSV)
    print(f"  {len(items)} gradable rows")

    name_index = build_filename_index(MESS_IMG)
    paths = []
    for it in items:
        stem = Path(it["image_id"]).stem
        if stem not in name_index:
            raise FileNotFoundError(f"Missing image for label {it['image_id']}")
        paths.append(name_index[stem])

    feats = extract_batch_features(model, paths, device, batch_size, "Messidor-2")
    assert not np.isnan(feats).any(), "NaNs in Messidor-2 DINOv2 features"
    assert feats.shape == (len(paths), 1024), f"shape {feats.shape} unexpected"

    grades = np.array([it["grade"] for it in items], dtype=np.int64)
    binary = (grades >= 2).astype(np.int64)
    ids = np.array([it["image_id"] for it in items])
    dme = np.array([it["dme"] for it in items], dtype=np.int64)

    np.save(MESS_OUT / "features.npy", feats)
    np.save(MESS_OUT / "grades.npy", grades)
    np.save(MESS_OUT / "binary_labels.npy", binary)
    np.save(MESS_OUT / "image_ids.npy", ids)
    np.save(MESS_OUT / "dme.npy", dme)
    print(f"  Saved -> {MESS_OUT}/  (grade dist {np.bincount(grades, minlength=5).tolist()}, referable {binary.mean()*100:.1f}%)")


# ---------------------------------------------------------------------------

def main() -> None:
    device = get_device()
    print(f"Device: {device}")
    model = load_dinov2(device)
    # MPS benefits from small batches on ViT-L to avoid allocation spikes;
    # 16 is a safe default that still saturates compute on M-series.
    batch_size = 16
    extract_aptos(model, device, batch_size)
    extract_messidor2(model, device, batch_size)
    print("\nAll DINOv2 caches written.")


if __name__ == "__main__":
    main()
