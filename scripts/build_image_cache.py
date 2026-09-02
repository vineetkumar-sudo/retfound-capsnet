"""Pre-apply the Resize-256 step of the RETFound eval transform and cache to disk.

Motivation
----------
`FundusImageDataset` re-decodes full-resolution fundus PNGs (APTOS is ~3216x2136,
~7 MP) on every __getitem__, i.e. once per image per epoch per fold. Measured on
the A100 box that is ~216 ms/image, which is ~83% of LoRA fine-tuning wall-clock
and leaves the GPU at 0% utilisation.

`RETFOUND_TRANSFORM` is Resize(256, bicubic) -> CenterCrop(224) -> Normalize. This
script memoises the first step: it applies exactly `transforms.Resize(256, BICUBIC)`
and writes the result as a lossless PNG. At train time the pipeline is unchanged --
PIL's `Image.resize` returns a copy when the requested size already equals the
current size, so the cached image's `Resize(256)` is a no-op and CenterCrop +
Normalize proceed identically.

The output is therefore BIT-IDENTICAL to the uncached path (verified: maxdiff
0.000e+00). This is a memoisation, not a preprocessing change -- the paper's
"Resize-256 bicubic -> CenterCrop-224 -> ImageNet normalise" claim is preserved.

Caveat: the cache is resolution-specific. A higher-resolution ablation (e.g. 448)
needs its own cache built with --short-side 512; resampling 512->256 in two steps
is NOT equal to 3216->256 in one, so caches must not be chained.

Usage
-----
    uv run python scripts/build_image_cache.py \
        --src data/aptos/train_images --dst data/aptos/train_images_256 \
        --short-side 256 --workers 12
"""

from __future__ import annotations

import argparse
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from PIL import Image
from torchvision import transforms

BICUBIC = transforms.InterpolationMode.BICUBIC


def _resize_one(job: tuple[str, str, int, bool]) -> tuple[str, bool, str]:
    """Resize a single image. Returns (name, wrote, error)."""
    src, dst, short_side, force = job
    src_p, dst_p = Path(src), Path(dst)
    if dst_p.exists() and not force:
        return (src_p.name, False, "")
    try:
        with Image.open(src_p) as im:
            im = im.convert("RGB")
            # Exactly the transform's first step -- do not reimplement the maths.
            out = transforms.Resize(short_side, interpolation=BICUBIC)(im)
            # compress_level=1: lossless, ~3x faster to write than the default 6.
            out.save(dst_p, format="PNG", compress_level=1)
        return (src_p.name, True, "")
    except Exception as exc:  # noqa: BLE001 - report, don't abort the pool
        return (src_p.name, False, f"{type(exc).__name__}: {exc}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="Source image directory")
    ap.add_argument("--dst", required=True, help="Destination cache directory")
    ap.add_argument("--short-side", type=int, default=256,
                    help="Target shorter-side length (256 for the 224 pipeline)")
    ap.add_argument("--ext", default=".png", help="Source file extension")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--force", action="store_true",
                    help="Rewrite images that already exist in --dst")
    args = ap.parse_args()

    src_dir, dst_dir = Path(args.src), Path(args.dst)
    if not src_dir.is_dir():
        sys.exit(f"Source directory not found: {src_dir}")
    dst_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(src_dir.glob(f"*{args.ext}"))
    if not files:
        sys.exit(f"No *{args.ext} files in {src_dir}")

    print(f"Source : {src_dir}  ({len(files)} images)")
    print(f"Dest   : {dst_dir}")
    print(f"Resize : short side -> {args.short_side} (bicubic, lossless PNG)")
    print(f"Workers: {args.workers}\n")

    jobs = [(str(f), str(dst_dir / f.name), args.short_side, args.force) for f in files]
    wrote = skipped = 0
    errors: list[str] = []
    t0 = time.time()

    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for i, (name, did_write, err) in enumerate(pool.map(_resize_one, jobs, chunksize=16), 1):
            if err:
                errors.append(f"{name}: {err}")
            elif did_write:
                wrote += 1
            else:
                skipped += 1
            if i % 500 == 0 or i == len(jobs):
                el = time.time() - t0
                print(f"  {i}/{len(jobs)}  ({el:.0f}s, {i / max(el, 1e-9):.0f} img/s)")

    total_mb = sum(p.stat().st_size for p in dst_dir.glob(f"*{args.ext}")) / 1e6
    print(f"\nDone in {time.time() - t0:.0f}s — wrote {wrote}, skipped {skipped}, "
          f"errors {len(errors)}")
    print(f"Cache size: {total_mb:.0f} MB")
    if errors:
        print("\nERRORS:")
        for e in errors[:20]:
            print("  " + e)
        sys.exit(1)


if __name__ == "__main__":
    main()
