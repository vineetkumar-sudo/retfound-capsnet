# Compute profile on a server GPU

Device: **NVIDIA A100-SXM4-40GB**, torch 2.11.0+cu130. Forward FLOPs are traced
with `torch.utils.flop_counter` (exact op-level counts, not analytic
estimates) and reported per image. Peak memory is
`torch.cuda.max_memory_allocated()` at the largest batch that fits.
Throughput is a training step for trainable tiers and a no-grad forward
for the frozen backbones.

| Component | Tier | Trainable | Total | FLOPs/img | Peak mem | Throughput |
|---|---|---:|---:|---:|---:|---:|
| Ordinal CapsNet head | frozen | 295,168 | 295,168 | 0.59 MFLOPs | 32 MiB | 9589 img/s (bs 64) |
| MLP + K-1 sigmoid head | frozen | 329,220 | 329,220 | 0.66 MFLOPs | 26 MiB | 28536 img/s (bs 64) |
| RETFound ViT-L/16 (frozen) | frozen | 0 | 304,326,632 | 123.11 GFLOPs | 1849 MiB | 144 img/s (bs 64) |
| DINOv2 ViT-L/14 (frozen) | frozen | 0 | 304,368,640 | 162.02 GFLOPs | 1963 MiB | 107 img/s (bs 64) |
| RETFound+LoRA r=8 + Ordinal head | lora | 1,081,600 | 304,383,232 | 123.42 GFLOPs | 15019 MiB | 65 img/s (bs 64) |
| RETFound full fine-tune + head | full-ft | 304,383,232 | 304,383,232 | 123.42 GFLOPs | 25293 MiB | 45 img/s (bs 64) |

## Batch-size sweep

| Component | Batch | ms/step | img/s | Peak mem |
|---|---:|---:|---:|---:|
| Ordinal CapsNet head | 1 | 6.53 | 153 | 23 MiB |
| Ordinal CapsNet head | 8 | 6.71 | 1191 | 23 MiB |
| Ordinal CapsNet head | 32 | 6.80 | 4707 | 27 MiB |
| Ordinal CapsNet head | 64 | 6.67 | 9589 | 32 MiB |
| MLP + K-1 sigmoid head | 1 | 2.17 | 460 | 26 MiB |
| MLP + K-1 sigmoid head | 8 | 2.26 | 3536 | 26 MiB |
| MLP + K-1 sigmoid head | 32 | 2.26 | 14164 | 26 MiB |
| MLP + K-1 sigmoid head | 64 | 2.24 | 28536 | 26 MiB |
| RETFound ViT-L/16 (frozen) | 1 | 13.96 | 72 | 1228 MiB |
| RETFound ViT-L/16 (frozen) | 8 | 64.49 | 124 | 1297 MiB |
| RETFound ViT-L/16 (frozen) | 32 | 231.53 | 138 | 1533 MiB |
| RETFound ViT-L/16 (frozen) | 64 | 445.28 | 144 | 1849 MiB |
| DINOv2 ViT-L/14 (frozen) | 1 | 19.04 | 53 | 1230 MiB |
| DINOv2 ViT-L/14 (frozen) | 8 | 89.90 | 89 | 1311 MiB |
| DINOv2 ViT-L/14 (frozen) | 32 | 303.81 | 105 | 1589 MiB |
| DINOv2 ViT-L/14 (frozen) | 64 | 598.28 | 107 | 1963 MiB |
| RETFound+LoRA r=8 + Ordinal head | 1 | 71.20 | 14 | 1446 MiB |
| RETFound+LoRA r=8 + Ordinal head | 8 | 136.89 | 58 | 2947 MiB |
| RETFound+LoRA r=8 + Ordinal head | 32 | 505.93 | 63 | 8106 MiB |
| RETFound+LoRA r=8 + Ordinal head | 64 | 978.47 | 65 | 15019 MiB |
| RETFound full fine-tune + head | 1 | 64.39 | 16 | 5833 MiB |
| RETFound full fine-tune + head | 8 | 213.03 | 38 | 7245 MiB |
| RETFound full fine-tune + head | 32 | 741.58 | 43 | 14961 MiB |
| RETFound full fine-tune + head | 64 | 1425.70 | 45 | 25293 MiB |
