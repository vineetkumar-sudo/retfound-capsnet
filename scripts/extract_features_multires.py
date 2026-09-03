"""Extract RETFound features at a chosen input resolution (reviewer R1-8/R2-3).

Reviewer comment 8: "Fundus image resizing to 224x224 pixels is performed
using standard bicubic interpolation. Such severe image resizing can destroy
subtle microvascular pathology (e.g., microaneurysms) ... and their influence
on feature extraction is not considered." Reviewer 2 comment 3 asks the same
thing: why 224, and how was the information loss assessed?

This answers it empirically rather than by argument: re-extract the frozen
RETFound features at a higher input resolution and re-run the identical head,
so the only variable is how many pixels the backbone sees.

RETFound is a ViT-L/16 pretrained at 224 (14x14 = 196 patches). Running it at
448 gives 28x28 = 784 patches, so the learned absolute position embedding must
be resampled to the new grid; `timm.layers.resample_abs_pos_embed` does this
with bicubic interpolation, the standard approach (ViT, DeiT, MAE all use it).
Patch-embedding weights are resolution-independent and are reused unchanged.

Preprocessing mirrors the 224 pipeline exactly at the new scale:
    Resize(S * 256/224, bicubic) -> CenterCrop(S) -> ImageNet normalise
so at S=224 it reduces to the paper's Resize-256 -> CenterCrop-224.

Usage:
    uv run python scripts/extract_features_multires.py --img-size 448 \
        --image-dir data/aptos/train_images_512 \
        --out-dir data/aptos/features_448
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np
import timm
import torch
from PIL import Image
from timm.layers import resample_abs_pos_embed
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils import enable_tf32

BICUBIC = transforms.InterpolationMode.BICUBIC


def build_transform(img_size: int) -> transforms.Compose:
    """The paper's transform, scaled to `img_size` (256/224 resize margin)."""
    resize_to = int(round(img_size * 256 / 224))
    return transforms.Compose([
        transforms.Resize(resize_to, interpolation=BICUBIC),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


class ImgDS(Dataset):
    def __init__(self, ids, image_dir: Path, tf, ext=".png"):
        self.ids, self.dir, self.tf, self.ext = list(ids), Path(image_dir), tf, ext

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        p = self.dir / f"{self.ids[i]}{self.ext}"
        return self.tf(Image.open(p).convert("RGB"))


def load_retfound_at(weights: str, img_size: int, device: torch.device):
    """RETFound ViT-L with the position embedding resampled to `img_size`."""
    model = timm.create_model("vit_large_patch16_224", pretrained=False,
                              img_size=img_size)
    ck = torch.load(weights, map_location="cpu", weights_only=False)
    sd = ck["model"] if "model" in ck else ck

    if "pos_embed" in sd:
        old = sd["pos_embed"]                       # (1, 1+196, 1024)
        want = model.pos_embed.shape[1]
        if old.shape[1] != want:
            n_prefix = old.shape[1] - 196           # cls (and any reg) tokens
            grid = int(round((want - n_prefix) ** 0.5))
            sd["pos_embed"] = resample_abs_pos_embed(
                old, new_size=[grid, grid], num_prefix_tokens=n_prefix,
                interpolation="bicubic", antialias=True,
            )
            print(f"  resampled pos_embed {tuple(old.shape)} -> "
                  f"{tuple(sd['pos_embed'].shape)} ({grid}x{grid} patches)")

    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"  loaded ({len(missing)} missing, {len(unexpected)} unexpected keys)")
    model.eval().to(device)
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--img-size", type=int, default=448)
    ap.add_argument("--image-dir", default="data/aptos/train_images_512")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--csv", default="data/aptos/train.csv")
    ap.add_argument("--weights", default="data/weights/RETFound_mae_natureCFP.pth")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--num-workers", type=int, default=8)
    args = ap.parse_args()

    out_dir = Path(args.out_dir or f"data/aptos/features_{args.img_size}")
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        enable_tf32()

    ids, labels = [], []
    with open(args.csv) as f:
        for row in csv.DictReader(f):
            ids.append(row["id_code"])
            labels.append(int(row["diagnosis"]))
    print(f"Device {device} | {len(ids)} images | img_size {args.img_size} "
          f"| images from {args.image_dir}")

    model = load_retfound_at(args.weights, args.img_size, device)
    tf = build_transform(args.img_size)
    dl = DataLoader(ImgDS(ids, args.image_dir, tf), batch_size=args.batch_size,
                    shuffle=False, num_workers=args.num_workers)

    feats, t0 = [], time.time()
    with torch.no_grad():
        for i, x in enumerate(dl, 1):
            f = model.forward_features(x.to(device))[:, 0]   # CLS
            feats.append(f.float().cpu().numpy())
            if i % 20 == 0 or i == len(dl):
                done = min(i * args.batch_size, len(ids))
                el = time.time() - t0
                print(f"  {done}/{len(ids)}  ({el:.0f}s, {done / max(el, 1e-9):.1f} img/s)")
    X = np.concatenate(feats).astype(np.float32)
    np.save(out_dir / "train_features.npy", X)
    np.save(out_dir / "train_labels.npy", np.array(labels, dtype=np.int64))
    np.save(out_dir / "train_ids.npy", np.array(ids))
    print(f"\nSaved {X.shape} -> {out_dir}  in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
