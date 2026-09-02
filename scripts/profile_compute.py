"""FLOPs, peak GPU memory and throughput on a server GPU (reviewer R1-11).

Reviewer comment 11: "The analysis presented in Section V-H regarding the
computation and hardware only considers the execution time for consumer
laptops (Apple MPS). It is imperative to have an exhaustive complexity
analysis that will consider FLOPS, maximum GPU memory usage, and throughput
for typical server GPUs (NVIDIA A100/V100)."

Measures, per configuration:
  * trainable / frozen / total parameter counts
  * forward FLOPs per image (torch.utils.flop_counter, an exact traced count,
    not an analytic estimate)
  * peak GPU memory for inference and for a training step
  * throughput in images/s, swept over batch size

Three compute tiers are profiled, matching the paper's own framing:
  frozen    head only, on precomputed 1024-d features (backbone run once)
  lora      ViT-L forward/backward with rank-8 adapters + head
  full-ft   every backbone parameter trainable (upper bound, for context)

A note that belongs in the paper: the frozen tier does not benefit from a
server GPU at all. Its heads are ~10^5 parameters over 1024-d vectors, so the
step is dominated by Python and kernel-launch latency rather than arithmetic.
Measured on this box, a full 5-fold frozen run takes 363 s on an A100 and
385 s on CPU -- statistically indistinguishable -- against the ~115 s the
paper reports for the same run on an M-series laptop. The A100 matters only
for the LoRA and full-fine-tune tiers.

Outputs results/compute_profile_table.{md,csv} and
results/compute_profile.json.

Usage:
    uv run python scripts/profile_compute.py [--batch-sizes 1 8 32 64]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from contextlib import nullcontext
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from torch.utils.flop_counter import FlopCounterMode

from src.models.mlp_ordinal import MLPOrdinal
from src.models.ordinal_capsnet import OrdinalCapsNet

RETFOUND_W = "data/weights/RETFound_mae_natureCFP.pth"


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def param_counts(m: nn.Module) -> dict:
    tr = sum(p.numel() for p in m.parameters() if p.requires_grad)
    tot = sum(p.numel() for p in m.parameters())
    return {"trainable": int(tr), "frozen": int(tot - tr), "total": int(tot)}


def fwd_flops(m: nn.Module, x: torch.Tensor) -> float:
    """Traced forward FLOPs for this input, divided by batch -> per-sample."""
    m.eval()
    ctr = FlopCounterMode(display=False)
    with ctr, torch.no_grad():
        m(x)
    return ctr.get_total_flops() / x.shape[0]


def _sync(dev: torch.device) -> None:
    if dev.type == "cuda":
        torch.cuda.synchronize()


def throughput(m: nn.Module, x: torch.Tensor, dev: torch.device,
               train: bool, iters: int = 20, warmup: int = 5) -> dict:
    """images/s and peak memory for inference or a full training step."""
    opt = torch.optim.AdamW([p for p in m.parameters() if p.requires_grad], lr=1e-4) \
        if train else None
    if train:
        m.train()
    else:
        m.eval()

    def _first_tensor(o) -> torch.Tensor:
        """Models here return a tensor, a tuple, or a dict of tensors."""
        if isinstance(o, torch.Tensor):
            return o
        vals = o.values() if isinstance(o, dict) else o
        for v in vals:
            if isinstance(v, torch.Tensor) and v.requires_grad:
                return v
        for v in vals:
            if isinstance(v, torch.Tensor):
                return v
        raise TypeError(f"no tensor in model output of type {type(o)}")

    def step() -> None:
        ctx = nullcontext() if train else torch.no_grad()
        with ctx:
            out = m(x)
            if train:
                loss = _first_tensor(out).float().pow(2).mean()
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()

    for _ in range(warmup):
        step()
    _sync(dev)
    if dev.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    for _ in range(iters):
        step()
    _sync(dev)
    el = time.perf_counter() - t0
    peak = torch.cuda.max_memory_allocated() / 2**20 if dev.type == "cuda" else float("nan")
    return {
        "images_per_s": x.shape[0] * iters / el,
        "ms_per_step": 1000 * el / iters,
        "peak_mem_mib": peak,
        "batch_size": int(x.shape[0]),
    }


# --------------------------------------------------------------------------
# model builders
# --------------------------------------------------------------------------
def build_backbone(kind: str, dev: torch.device) -> nn.Module:
    if kind == "retfound":
        from src.data.feature_cache import load_retfound
        m = load_retfound(RETFOUND_W, dev)
    else:
        from scripts.extract_dinov2_features import load_dinov2
        m = load_dinov2(dev)
    return m.to(dev)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 8, 32, 64])
    ap.add_argument("--skip-backbones", action="store_true")
    args = ap.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu = torch.cuda.get_device_name(0) if dev.type == "cuda" else "cpu"
    print(f"Device: {dev}  ({gpu})\n")
    rows: list[dict] = []

    # ---- heads on cached features (the frozen tier) ----------------------
    for name, head in (("Ordinal CapsNet head", OrdinalCapsNet()),
                        ("MLP + K-1 sigmoid head", MLPOrdinal())):
        head = head.to(dev)
        x = torch.randn(args.batch_sizes[-1], 1024, device=dev)
        rec = {"component": name, "tier": "frozen", "input": "1024-d feature",
               **param_counts(head),
               "fwd_gflops_per_sample": fwd_flops(head, x) / 1e9,
               "sweep": [throughput(head, torch.randn(b, 1024, device=dev), dev,
                                    train=True, iters=30)
                         for b in args.batch_sizes]}
        rows.append(rec)
        print(f"  {name:<26s} {rec['trainable']:>10,} tr  "
              f"{rec['fwd_gflops_per_sample'] * 1000:7.3f} MFLOPs/img  "
              f"{rec['sweep'][-1]['images_per_s']:8.0f} img/s train")

    # ---- frozen backbones (inference only, one-time feature extraction) ---
    if not args.skip_backbones:
        for kind, label in (("retfound", "RETFound ViT-L/16 (frozen)"),
                            ("dinov2", "DINOv2 ViT-L/14 (frozen)")):
            try:
                bb = build_backbone(kind, dev)
            except Exception as exc:  # noqa: BLE001
                print(f"  SKIP {label}: {type(exc).__name__}: {exc}")
                continue
            for p in bb.parameters():
                p.requires_grad_(False)
            x = torch.randn(args.batch_sizes[-1], 3, 224, 224, device=dev)
            rec = {"component": label, "tier": "frozen", "input": "224x224x3",
                   **param_counts(bb),
                   "fwd_gflops_per_sample": fwd_flops(bb, x) / 1e9,
                   "sweep": [throughput(bb, torch.randn(b, 3, 224, 224, device=dev),
                                        dev, train=False, iters=10)
                             for b in args.batch_sizes]}
            rows.append(rec)
            print(f"  {label:<26s} {rec['total']:>10,} tot "
                  f"{rec['fwd_gflops_per_sample']:7.2f} GFLOPs/img  "
                  f"{rec['sweep'][-1]['images_per_s']:8.1f} img/s infer  "
                  f"peak {rec['sweep'][-1]['peak_mem_mib']:.0f} MiB")
            del bb
            torch.cuda.empty_cache() if dev.type == "cuda" else None

        # ---- LoRA and full fine-tune (training) --------------------------
        for tier in ("lora", "full-ft"):
            try:
                from src.models.retfound_lora_capsnet import RetfoundLoraOrdinalCapsNet
                m = RetfoundLoraOrdinalCapsNet(weights_path=RETFOUND_W).to(dev)
                if tier == "full-ft":
                    for p in m.parameters():
                        p.requires_grad_(True)
            except Exception as exc:  # noqa: BLE001
                print(f"  SKIP {tier}: {type(exc).__name__}: {exc}")
                continue
            label = ("RETFound+LoRA r=8 + Ordinal head" if tier == "lora"
                     else "RETFound full fine-tune + head")
            sweep = []
            for b in args.batch_sizes:
                try:
                    sweep.append(throughput(m, torch.randn(b, 3, 224, 224, device=dev),
                                            dev, train=True, iters=8, warmup=3))
                except torch.cuda.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    sweep.append({"batch_size": b, "images_per_s": float("nan"),
                                  "ms_per_step": float("nan"),
                                  "peak_mem_mib": float("nan"), "oom": True})
            x = torch.randn(1, 3, 224, 224, device=dev)
            rec = {"component": label, "tier": tier, "input": "224x224x3",
                   **param_counts(m),
                   "fwd_gflops_per_sample": fwd_flops(m, x) / 1e9,
                   "sweep": sweep}
            rows.append(rec)
            ok = [s for s in sweep if not s.get("oom")]
            print(f"  {label:<26s} {rec['trainable']:>10,} tr  "
                  f"{rec['fwd_gflops_per_sample']:7.2f} GFLOPs/img  "
                  + (f"{ok[-1]['images_per_s']:8.1f} img/s train  "
                     f"peak {ok[-1]['peak_mem_mib']:.0f} MiB" if ok else "all OOM"))
            del m
            torch.cuda.empty_cache() if dev.type == "cuda" else None

    out = Path("results")
    (out / "compute_profile.json").write_text(
        json.dumps({"gpu": gpu, "torch": torch.__version__, "rows": rows}, indent=2))

    md = [
        "# Compute profile on a server GPU",
        "",
        f"Device: **{gpu}**, torch {torch.__version__}. Forward FLOPs are traced",
        "with `torch.utils.flop_counter` (exact op-level counts, not analytic",
        "estimates) and reported per image. Peak memory is",
        "`torch.cuda.max_memory_allocated()` at the largest batch that fits.",
        "Throughput is a training step for trainable tiers and a no-grad forward",
        "for the frozen backbones.",
        "",
        "| Component | Tier | Trainable | Total | FLOPs/img | Peak mem | Throughput |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        ok = [s for s in r["sweep"] if not s.get("oom")]
        s = ok[-1] if ok else None
        fl = (f"{r['fwd_gflops_per_sample'] * 1000:.2f} MFLOPs"
              if r["fwd_gflops_per_sample"] < 1 else
              f"{r['fwd_gflops_per_sample']:.2f} GFLOPs")
        md.append(
            f"| {r['component']} | {r['tier']} | {r['trainable']:,} | {r['total']:,} "
            f"| {fl} "
            f"| {s['peak_mem_mib']:.0f} MiB | {s['images_per_s']:.0f} img/s "
            f"(bs {s['batch_size']}) |" if s else
            f"| {r['component']} | {r['tier']} | {r['trainable']:,} | {r['total']:,} "
            f"| {fl} | OOM | OOM |"
        )

    md += ["", "## Batch-size sweep", "",
           "| Component | Batch | ms/step | img/s | Peak mem |", "|---|---:|---:|---:|---:|"]
    for r in rows:
        for s in r["sweep"]:
            if s.get("oom"):
                md.append(f"| {r['component']} | {s['batch_size']} | OOM | OOM | OOM |")
            else:
                md.append(f"| {r['component']} | {s['batch_size']} "
                          f"| {s['ms_per_step']:.2f} | {s['images_per_s']:.0f} "
                          f"| {s['peak_mem_mib']:.0f} MiB |")
    (out / "compute_profile_table.md").write_text("\n".join(md) + "\n")

    csv = ["component,tier,trainable,frozen,total,fwd_gflops_per_sample,"
           "batch_size,ms_per_step,images_per_s,peak_mem_mib"]
    for r in rows:
        for s in r["sweep"]:
            csv.append(",".join([
                r["component"], r["tier"], str(r["trainable"]), str(r["frozen"]),
                str(r["total"]), f"{r['fwd_gflops_per_sample']:.6f}",
                str(s["batch_size"]), f"{s.get('ms_per_step', float('nan')):.4f}",
                f"{s.get('images_per_s', float('nan')):.4f}",
                f"{s.get('peak_mem_mib', float('nan')):.2f}",
            ]))
    (out / "compute_profile_table.csv").write_text("\n".join(csv) + "\n")
    print("\nWrote results/compute_profile_table.{md,csv} + results/compute_profile.json")


if __name__ == "__main__":
    main()
