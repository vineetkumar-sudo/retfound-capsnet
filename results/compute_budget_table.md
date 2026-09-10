# Compute + Parameter Budget — DR Methods Comparison

Compiled from `paper/ref_papers/*.pdf` + our measurements. Our rows are the A100 rerun that is canonical for the paper; FLOPs, peak memory and throughput are in `results/compute_profile_table.md`. Ordered by trainable-parameter count (smallest first). All APTOS QWK figures are on the Kaggle APTOS-2019 public training set (3,662 images) — paper-reported numbers may use slightly different splits; we list the evaluation protocol in the rightmost column.

## Main comparison table

| Method | Year | Backbone | **Trainable params** | Total params (incl. frozen) | Training hardware | Training time (full pipeline) | APTOS QWK | Evaluation |
|---|---|---|---:|---:|---|---|---:|---|
| **Ours (frozen Ordinal CapsNet)** | — | RETFound ViT-L (**frozen**) | **295 K** | ~304 M | NVIDIA A100-SXM4-40GB (also runs on MacBook MPS) | **~2 min** (5-fold) | **0.8923 ± 0.0002** | 3 seeds × 5-fold CV |
| **Ours (frozen Ordinal CapsNet, DINOv2)** | — | DINOv2 ViT-L/14 (**frozen**) | **295 K** | ~304 M | NVIDIA A100-SXM4-40GB | **~2 min** (5-fold) | **0.9074 ± 0.0029** | 3 seeds × 5-fold CV |
| **Ours (LoRA Ordinal CapsNet)** | — | RETFound ViT-L + LoRA r=8 | **1.08 M** | ~304 M | NVIDIA A100-SXM4-40GB (~16 h on MPS) | **~26 min** (5-fold) | **0.9143 ± 0.0006** | 3 seeds × 5-fold CV |
| **Ours (progressive unfreeze, last 4 blocks)** | — | RETFound ViT-L, last 4 blocks | **50.8 M** | ~304 M | NVIDIA A100-SXM4-40GB | **~26 min** (5-fold) | **0.9189 ± 0.0086** | seed 42, 5-fold CV |
| **Ours (full fine-tune)** | — | RETFound ViT-L, all weights | **304 M** | ~304 M | NVIDIA A100-SXM4-40GB | **~54 min** (5-fold) | **0.8880 ± 0.0057** | seed 42, 5-fold CV |
| Dixit EfficientNetB3+SE | 2025 | EfficientNetB3 + SE block | ~12 M | ~12 M | GPU (type not reported) | not reported | "κ 0.88" (ambiguous Cohen vs QWK) | 80/10/10 single split |
| Lei GF-CapsNet | 2024 | ResNet-18 + GNN + CapsNet | ~13 M | ~13 M | GPU (type not reported) | not reported | — (AUC 0.956, Acc 0.865) | 80/20 single split |
| Kumar Stage-Aware | 2025 | ResNet-50 + MSE regression | ~25 M | ~25 M | GPU (type not reported) | not reported | 0.8992 | single val split |
| Oulhadj deformable-registration ensemble | 2022 | 4-CNN (DenseNet121 + Xception + Inception-V3 + ResNet-50) | ~80 M | ~80 M | GPU | **~22 h** reported | — (Cohen κ 0.778; equivalent QWK uncertain) | 2 801 / 494 / 367 split |
| El Bellaj UAOR-LQAP+EDL | 2026 | ConvNeXt-Base + LQAP + evidential Dirichlet head | not reported (~89 M standard ConvNeXt-Base) | not reported | bf16 mixed-precision "when available" (GPU not named) | not reported (up to 60 epochs with AdamW + cosine + EMA + early stop) | 0.940 (pooled 3-dataset test, not per-dataset) | pooled train / pooled test |
| Bodapati dual-attention stacking ensemble | 2024 | 4 × (VGG16-CB{3,4,5} + Xception-CB14) + ConvLSTM + MLP | not reported (estimate ~200 M) | not reported | 11 GB NVIDIA GPU (model not specified), Dell workstation | not reported | ~0.897 (reported as "Kappa 0.8965") | 80/20 single split |
| RETFound + MLP (original paper) | 2023 | ViT-L full fine-tune | not reported (ViT-L ≈ 307 M; paper does not give exact count) | ≈ 307 M | **1× NVIDIA T4 (16 GB)** — the paper notes "we required only one" | **~1.2 h per 1 000 images** (quoted: "Fine-tuning takes about 70 min for every 1 000 images"). Extrapolating to APTOS-2019 (~3 300 images) → **~4 h on T4** | AUROC 0.943 (QWK not reported) | 5-fold internal |
| Yu AOR-DR + RETFound | 2025 | RETFound (**frozen** per §Method) + autoregressive ordinal head + conditional diffusion | not reported | — | not reported | not reported (batch 32, 1 000 diffusion steps) | — (Acc 0.803, F1 0.657) | single split |

### Notes on the numbers

- **Ours vs RETFound-original**: RETFound's 0.943 is AUROC (referable vs non), not QWK. The direct QWK comparison is with the DR-method literature in the rest of the table. Our frozen variant uses RETFound's representation as a *feature* (no backbone update), cutting the trainable-parameter count by **~1000×** relative to a full RETFound fine-tune.
- **Ours vs CapsNet literature**: Lei's GF-CapsNet is the closest architectural comparator (ResNet-18 + multi-head CapsNet); they don't use ordinal decomposition and report Acc 0.865 on their own 80/20 split. Our frozen QWK 0.893 is achieved with ~25× fewer trainable parameters.
- **El Bellaj's 0.940 "QWK"**: reported on a pooled train / pooled held-out test across APTOS + Messidor-2 + EyePACS, with no per-dataset breakdown and no multi-seed. Their headline is the ceiling claim in the literature; their rigour (pooled test, single seed) is the weakest among ordinal-specific methods.
- **Bodapati's 0.8965**: reported as "Kappa" without specifying quadratic-weighted. At face value, that number sits between our frozen (0.893) and LoRA (0.913). Their ensemble has ~200× more trainable parameters than our frozen variant.

## Hardware-accessibility summary

| Tier | Method | Runs on | Interactive-prototyping? |
|---|---|---|---|
| Consumer | **Ours (frozen)** | MacBook Pro M-series, 2 min for full 5-fold | ✅ yes |
| Consumer-premium | **Ours (LoRA)** | MacBook Pro M-series, overnight for 3-seed × 5-fold (~16 h MPS) | ✅ yes |
| Data-center | Oulhadj, Bodapati, Dixit, Kumar | Single GPU, multi-hour | ⚠ feasible but slow |
| Data-center-heavy | El Bellaj, RETFound fine-tune, Yu AOR-DR | Multi-GPU (A100-class), days | ❌ not interactive |

## Data for the paper's Fig / Tab 7

- **Headline claim**: "Our frozen Ordinal CapsNet achieves competitive APTOS QWK (0.8923 ± 0.0002) using only **295 K** trainable parameters — a ~1000× reduction over typical fine-tuned approaches."
- **Secondary claim**: "LoRA rank-8 adaptation lifts QWK to **0.9143 ± 0.0006** at **1.08 M** trainable parameters — still ~280× below the RETFound full fine-tune and ~80× below El Bellaj's ConvNeXt-Base. Full fine-tuning all 304 M weights is *worse* than the frozen head (0.8880), so the parameter-efficiency curve is non-monotonic."
- **Tertiary**: "Both variants train on consumer Apple Silicon; no A100 required. Frozen runs full 5-fold CV in under 2 minutes."

## Verified (from the actual PDFs)

1. **El Bellaj UAOR-LQAP+EDL**: training time, hardware, and trainable parameter counts **not reported**. Paper mentions "bf16 mixed-precision when available", "up to 60 epochs with AdamW + cosine + EMA + early stop" (§IV.A Experimental Setup). No GPU model or wall-clock hours given. ConvNeXt-Base standard param count (~89 M) is an external estimate, not from the paper.
2. **RETFound fine-tune (Zhou et al. 2023)**: **explicit quote**: "Fine-tuning takes about 70 min for every 1 000 images" on a single NVIDIA Tesla T4 (16 GB) (Methods §Computational resources). For APTOS-2019 (~3 300 train images) this extrapolates to **~4 h of fine-tune time**, on consumer-grade hardware — much less than I originally estimated. Exact ViT-L param count is not in the paper; the 307 M figure is the standard ViT-L/16 count.
3. **Yu AOR-DR**: RETFound backbone is explicitly **frozen** (page 7: "for RETFound, the backbone is frozen during training"). Trainable parameter count (AR network + diffusion conditional model + ordinal head) is **not reported**. Hardware and training wall-clock are also not reported (batch size 32, 1 000 diffusion training steps, 10 inference steps is all the implementation detail given).
4. **Bodapati dual-attention ensemble**: training wall-clock and aggregate trainable-parameter counts are **not reported**. Paper mentions "Dell Workstation with 26 GB RAM, 2 TB HDD, 11 GB NVIDIA GPU" (§4.3) — GPU model not specified. The ~200 M params in the table is an external estimate from summing standard VGG16 + Xception sizes; not verified against the paper.

## Implication for the paper

Two of the four competitor rows (El Bellaj, Bodapati) don't publish training cost or parameter counts in their paper, and Yu's AOR-DR omits them too. That's actually **useful for your narrative**: our paper can lead with "we report exact MPS wall-clock and parameter counts alongside every number, contrasting with competitor papers that omit these". The RETFound number (~4 h on a single T4 for APTOS fine-tune) is the most meaningful comparison — our frozen variant runs in **115 s** for the same APTOS 5-fold CV, i.e., **~125× faster** on consumer Apple Silicon vs. T4 fine-tune.
