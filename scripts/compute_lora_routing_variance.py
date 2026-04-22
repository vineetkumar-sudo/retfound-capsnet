"""Post-hoc: compute per-sample routing variance for each LoRA fold.

The LoRA training script saves `head_lengths` to `preds_fold{N}.npz` but not
`routing_variance` (we don't pass `return_routing_history=True` during eval).
This script loads each fold's saved weights, replays the val loader with
routing history enabled, and writes `routing_variance_fold{N}.npy` alongside
the existing preds so that the LoRA accuracy-coverage figure can plot all
three UQ curves.

Walks every `results/lora_ordinal_capsnet/seed{S}/` subdir (multi-seed
layout), falling back to the flat top-level layout when no seed subdirs
exist. Existing `routing_variance_fold{N}.npy` files are left untouched
(delete them to recompute).

Usage:
    uv run python scripts/compute_lora_routing_variance.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, ".")

import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold, train_test_split
from torch.utils.data import DataLoader

from src.data.image_dataset import FundusImageDataset
from src.models.retfound_lora_capsnet import RetfoundLoraOrdinalCapsNet


APTOS_CSV = Path("data/aptos/train.csv")
APTOS_IMG = Path("data/aptos/train_images")
OUT_DIR = Path("results/lora_ordinal_capsnet")

SEED = 42
HOLDOUT_FRAC = 0.10
N_FOLDS = 5
BATCH_SIZE = 16  # eval only — no grad


def get_device() -> torch.device:
    if torch.cuda.is_available(): return torch.device("cuda")
    if torch.backends.mps.is_available(): return torch.device("mps")
    return torch.device("cpu")


def load_aptos_split() -> tuple[np.ndarray, np.ndarray]:
    import csv
    ids: list[str] = []
    labels: list[int] = []
    with APTOS_CSV.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            ids.append(row["id_code"])
            labels.append(int(row["diagnosis"]))
    ids_arr = np.array(ids)
    labels_arr = np.array(labels, dtype=np.int64)
    pool_idx, _ = train_test_split(
        np.arange(len(ids_arr)), test_size=HOLDOUT_FRAC,
        stratify=labels_arr, random_state=SEED,
    )
    return ids_arr[pool_idx], labels_arr[pool_idx]


@torch.no_grad()
def compute_routing_variance_for_fold(model: RetfoundLoraOrdinalCapsNet,
                                      loader: DataLoader,
                                      device: torch.device) -> np.ndarray:
    """Per-sample routing variance in val-loader order (matches preds_fold{N}.npz)."""
    model.eval()
    out = []
    for X, _ in loader:
        X = X.to(device)
        rh = model(X, return_routing_history=True)["routing_history"]
        # Shape: (H, R, B, P, 2). std over R, mean over H/P/2 → (B,)
        rv = rh.std(dim=1).mean(dim=(0, 2, 3)).cpu().numpy().astype(np.float32)
        out.append(rv)
    return np.concatenate(out, axis=0)


def compute_for_dir(seed_dir: Path, splits: list, ids_pool, y_pool,
                    device: torch.device, label: str) -> int:
    """Walk weights_fold*.pt in seed_dir, write routing_variance_fold*.npy.
    Returns the number of folds computed this call."""
    n_done = 0
    for fold_idx, (_, va_idx) in enumerate(splits):
        weights_path = seed_dir / f"weights_fold{fold_idx + 1}.pt"
        rv_path = seed_dir / f"routing_variance_fold{fold_idx + 1}.npy"
        if not weights_path.exists():
            print(f"  {label} fold {fold_idx + 1}: {weights_path.name} missing — SKIP")
            continue
        if rv_path.exists():
            print(f"  {label} fold {fold_idx + 1}: {rv_path.name} already present — SKIP "
                  f"(delete to recompute)")
            continue

        torch.manual_seed(SEED + fold_idx)
        np.random.seed(SEED + fold_idx)

        ds = FundusImageDataset(ids_pool[va_idx], y_pool[va_idx],
                                APTOS_IMG, image_ext=".png")
        loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=2, pin_memory=False)

        model = RetfoundLoraOrdinalCapsNet().to(device)
        state = torch.load(weights_path, map_location=device, weights_only=True)
        model.load_state_dict(state, strict=False)
        loaded = set(state.keys())
        assert any("lora_" in k for k in loaded), "No LoRA keys in saved weights!"
        assert any(k.startswith("head.") for k in loaded), "No head keys in saved weights!"

        t0 = time.time()
        rv = compute_routing_variance_for_fold(model, loader, device)
        np.save(rv_path, rv)
        print(f"  {label} fold {fold_idx + 1}: routing_variance saved "
              f"({rv.shape}, mean={rv.mean():.4f})  ({time.time() - t0:.1f}s)")
        n_done += 1
    return n_done


def main() -> None:
    device = get_device()
    print(f"Device: {device}")

    ids_pool, y_pool = load_aptos_split()
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    splits = list(skf.split(ids_pool, y_pool))

    seed_dirs = sorted(OUT_DIR.glob("seed*"))
    if seed_dirs:
        print(f"Multi-seed layout: {len(seed_dirs)} seed dir(s)\n")
        for sd in seed_dirs:
            print(f"=== {sd} ===")
            compute_for_dir(sd, splits, ids_pool, y_pool, device, label=sd.name)
    else:
        print(f"Single-seed flat layout at {OUT_DIR}\n")
        compute_for_dir(OUT_DIR, splits, ids_pool, y_pool, device, label="flat")


if __name__ == "__main__":
    main()
