"""Extract and cache RETFound CLS token features for all images.

Idempotent: clears any existing cached features and re-extracts from scratch.
Run: uv run python -m src.data.feature_cache
"""

import csv
import shutil
import time
from pathlib import Path

import numpy as np
import timm
import torch
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

# Official RETFound eval transform (matches util/datasets.py in RETFound_MAE repo)
RETFOUND_TRANSFORM = transforms.Compose([
    transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


def load_retfound(weights_path: str, device: torch.device) -> torch.nn.Module:
    """Load frozen RETFound ViT-L and move to device."""
    model = timm.create_model("vit_large_patch16_224", pretrained=False)
    checkpoint = torch.load(weights_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint["model"] if "model" in checkpoint else checkpoint
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    model.to(device)
    for p in model.parameters():
        p.requires_grad = False
    return model


@torch.no_grad()
def extract_features(
    model: torch.nn.Module,
    image_dir: Path,
    image_ids: list[str],
    device: torch.device,
    batch_size: int = 32,
    split_name: str = "",
) -> np.ndarray:
    """Extract 1024-dim CLS token features for a list of image IDs.

    Returns:
        np.ndarray of shape (N, 1024), float32.
    """
    all_features = []
    total_batches = (len(image_ids) + batch_size - 1) // batch_size
    t0 = time.time()

    for batch_idx, i in enumerate(
        tqdm(
            range(0, len(image_ids), batch_size),
            desc=f"  {split_name}",
            unit="batch",
        )
    ):
        batch_ids = image_ids[i : i + batch_size]
        batch_tensors = []
        for img_id in batch_ids:
            img_path = image_dir / f"{img_id}.png"
            img = Image.open(img_path).convert("RGB")
            batch_tensors.append(RETFOUND_TRANSFORM(img))

        batch = torch.stack(batch_tensors).to(device)
        features = model.forward_features(batch)
        cls_tokens = features[:, 0]  # (B, 1024)
        all_features.append(cls_tokens.cpu().numpy())

        # Print ETA every 20 batches
        if (batch_idx + 1) % 20 == 0:
            elapsed = time.time() - t0
            rate = (batch_idx + 1) / elapsed
            remaining = (total_batches - batch_idx - 1) / rate
            imgs_done = min((batch_idx + 1) * batch_size, len(image_ids))
            print(
                f"    [{imgs_done}/{len(image_ids)} images] "
                f"{elapsed:.0f}s elapsed, ~{remaining:.0f}s remaining"
            )

    elapsed = time.time() - t0
    print(f"    Done: {len(image_ids)} images in {elapsed:.1f}s "
          f"({len(image_ids) / elapsed:.1f} img/s)")

    return np.concatenate(all_features, axis=0)


def cache_features(
    weights_path: str = "data/weights/RETFound_mae_natureCFP.pth",
    data_dir: str = "data/aptos",
    output_dir: str = "data/aptos/features",
    batch_size: int = 32,
    device_name: str | None = None,
) -> None:
    """Run the full feature extraction pipeline and save to .npy files.

    Idempotent — wipes output_dir before extracting so re-runs are clean.

    Saves:
        {output_dir}/train_features.npy  — (N_train, 1024) float32
        {output_dir}/train_labels.npy    — (N_train,) int64
        {output_dir}/train_ids.npy       — (N_train,) str
        {output_dir}/test_features.npy   — (N_test, 1024) float32
        {output_dir}/test_ids.npy        — (N_test,) str
    """
    data_path = Path(data_dir)
    out_path = Path(output_dir)

    # Idempotent: clear previous run
    if out_path.exists():
        print(f"Clearing previous cache at {out_path}/")
        shutil.rmtree(out_path)
    out_path.mkdir(parents=True, exist_ok=True)

    # Pick device
    if device_name:
        device = torch.device(device_name)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}")

    # Load model
    print("Loading RETFound weights...")
    t0 = time.time()
    model = load_retfound(weights_path, device)
    print(f"Model loaded in {time.time() - t0:.1f}s\n")

    # --- Train split ---
    print("=== Train split ===")
    train_ids, train_labels = [], []
    with open(data_path / "train.csv") as f:
        reader = csv.DictReader(f)
        for row in reader:
            train_ids.append(row["id_code"])
            train_labels.append(int(row["diagnosis"]))
    print(f"  Found {len(train_ids)} images")

    train_features = extract_features(
        model, data_path / "train_images", train_ids, device, batch_size,
        split_name="Train",
    )
    train_labels_arr = np.array(train_labels, dtype=np.int64)

    np.save(out_path / "train_features.npy", train_features)
    np.save(out_path / "train_labels.npy", train_labels_arr)
    np.save(out_path / "train_ids.npy", np.array(train_ids))
    print(f"  Saved: train_features {train_features.shape}, "
          f"train_labels {train_labels_arr.shape}\n")

    # --- Test split ---
    print("=== Test split ===")
    test_ids = []
    with open(data_path / "test.csv") as f:
        reader = csv.DictReader(f)
        for row in reader:
            test_ids.append(row["id_code"])
    print(f"  Found {len(test_ids)} images")

    test_features = extract_features(
        model, data_path / "test_images", test_ids, device, batch_size,
        split_name="Test",
    )

    np.save(out_path / "test_features.npy", test_features)
    np.save(out_path / "test_ids.npy", np.array(test_ids))
    print(f"  Saved: test_features {test_features.shape}\n")

    # --- Summary ---
    total_bytes = sum(f.stat().st_size for f in out_path.glob("*.npy"))
    print("=" * 40)
    print(f"All features saved to {out_path}/")
    print(f"Total cache size: {total_bytes / 1024 / 1024:.1f} MB")
    print(f"Files:")
    for f in sorted(out_path.glob("*.npy")):
        arr = np.load(f, mmap_mode="r")
        print(f"  {f.name:30s} shape={str(arr.shape):20s} dtype={arr.dtype}")


if __name__ == "__main__":
    cache_features()
