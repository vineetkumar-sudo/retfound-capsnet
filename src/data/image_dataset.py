"""Raw-image PyTorch Dataset used by LoRA fine-tuning (backbone sees gradients).

For cached-feature training we use `np.load(...)` directly; for LoRA fine-tuning
we need the images themselves so they flow through the ViT on every step. This
dataset mirrors the official RETFound eval pipeline (Resize 256 bicubic → CenterCrop 224
→ ImageNet normalize) and supports both APTOS (.png) and IDRiD (.jpg) layouts.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from src.data.feature_cache import RETFOUND_TRANSFORM


class FundusImageDataset(Dataset):
    """Load fundus images from disk with RETFound eval preprocessing applied."""

    def __init__(
        self,
        image_ids: list[str] | np.ndarray,
        labels: list[int] | np.ndarray,
        image_dir: Path | str,
        image_ext: str = ".png",
        label_dtype: torch.dtype = torch.long,
    ):
        self.image_ids = [str(x) for x in image_ids]
        self.labels = np.asarray(labels)
        self.image_dir = Path(image_dir)
        self.image_ext = image_ext
        self.label_dtype = label_dtype

    def __len__(self) -> int:
        return len(self.image_ids)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        img_id = self.image_ids[idx]
        path = self.image_dir / f"{img_id}{self.image_ext}"
        img = Image.open(path).convert("RGB")
        x = RETFOUND_TRANSFORM(img)  # (3, 224, 224) float32 normalised
        y = torch.tensor(self.labels[idx], dtype=self.label_dtype)
        return x, y
