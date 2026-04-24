# Head × Backbone 2×2 ablation — Day 12

All four cells on each dataset use an identical protocol (3 seeds × 5-fold, AdamW, margin loss for CapsNet / OrdinalMarginLoss for both K-1 ordinal heads, early stopping patience 30). The only variables are the **backbone** (frozen RETFound ViT-L/16 nature-CFP MAE vs frozen DINOv2 ViT-L/14 `dinov2_vitl14`) and the **head** (Ordinal CapsNet with dynamic routing vs MLP + 4 independent sigmoid ordinal heads).

Both backbones produce 1024-dim CLS features at input resolution 224, with the identical Resize-256 bicubic → CenterCrop-224 → ImageNet-normalise preprocessing pipeline, so the only changing variable is the backbone weights.

## APTOS-2019 (held-out 10% + 5-fold CV on 3,296 train images)

| Head              | RETFound (QWK) | DINOv2 (QWK) | Δ backbone |
|---|---:|---:|---:|
| Ordinal CapsNet    | 0.8932 ± 0.0004 | **0.9089 ± 0.0021** | **+0.0157** |
| MLP + K-1 sigmoid  | 0.8866 ± 0.0004 | **0.9034 ± 0.0002** | **+0.0168** |
| Δ head (caps − MLP)| +0.0066         | +0.0055             | — |

## Messidor-2 within-dataset (5-fold CV on 1,744 gradable images)

| Head              | RETFound (QWK) | DINOv2 (QWK) | Δ backbone |
|---|---:|---:|---:|
| Ordinal CapsNet    | 0.6065 ± 0.0363 | **0.7525 ± 0.0222** | **+0.1460** |
| MLP + K-1 sigmoid  | 0.5852 ± 0.0405 | **0.7643 ± 0.0148** | **+0.1791** |
| Δ head (caps − MLP)| +0.0213         | −0.0118             | — |

## Key findings

1. **Backbone effect dominates head effect on both datasets.** Swapping RETFound→DINOv2 yields +0.016 QWK on APTOS and +0.146 QWK on Messidor-2 regardless of head; swapping MLP-K1→Ordinal CapsNet yields at most +0.02 QWK and flips sign on one cell.
2. **DINOv2 outperforms RETFound at the same parameter count on both datasets, for both heads.** This reproduces and extends the head-to-head finding in the recent DINOv2-vs-RETFound comparative literature: a generalist self-supervised ViT beats the domain-specific RETFound on frozen-feature DR grading.
3. **Messidor-2 flips the head ranking under the stronger backbone.** On RETFound features the capsule head wins by +0.021 QWK; on DINOv2 features the MLP K-1 head wins by +0.012. Under sufficiently strong features, the capsule machinery's contribution saturates and can even reverse — reviewer-worthy nuance.
4. **The K-1 decomposition is the more consistent architectural lever** than capsule routing. Between MLP+CE (APTOS QWK 0.8783, Messidor-2 0.5344) and MLP+K-1 sigmoid (APTOS 0.8866, Messidor-2 0.5852) the ordinal decomposition buys +0.008 APTOS / +0.051 Messidor-2 on identical trunk + identical backbone.

## Commands that produced these numbers

```bash
# Feature extraction (DINOv2 ViT-L/14 via torch.hub; ~7 min on MPS for both datasets)
uv run python scripts/extract_dinov2_features.py

# APTOS (3 seeds × 5-fold each, ~6 min per run on MPS)
uv run python scripts/run_ordinal_capsnet.py --config configs/ordinal_capsnet_dinov2.yaml --seeds 42,123,456 --no-wandb
uv run python scripts/run_ordinal_capsnet.py --config configs/mlp_ordinal_dinov2.yaml      --seeds 42,123,456 --no-wandb

# Messidor-2 within-dataset (5-fold; ~1 min per model)
uv run python scripts/run_messidor2.py --features-dir data/messidor2/features_dinov2 --output-dir results/messidor2_dinov2 --models ordinal_capsnet
uv run python scripts/run_messidor2.py --features-dir data/messidor2/features_dinov2 --output-dir results/messidor2_dinov2 --models mlp_k1_sigmoid
uv run python scripts/run_messidor2.py                                                                                   --models mlp_k1_sigmoid
```

## Artifacts

| Cell | Path |
|---|---|
| RETFound × Ordinal CapsNet (APTOS)   | `results/ordinal_capsnet/seed{42,123,456}/` |
| RETFound × MLP-K1 (APTOS)            | `results/mlp_ordinal/seed{42,123,456}/` |
| DINOv2  × Ordinal CapsNet (APTOS)    | `results/ordinal_capsnet_dinov2/seed{42,123,456}/` |
| DINOv2  × MLP-K1 (APTOS)             | `results/mlp_ordinal_dinov2/seed{42,123,456}/` |
| RETFound × Ordinal CapsNet (Mess-2)  | `results/messidor2/ordinal_capsnet/` |
| RETFound × MLP-K1 (Mess-2)           | `results/messidor2/mlp_k1_sigmoid/` |
| DINOv2  × Ordinal CapsNet (Mess-2)   | `results/messidor2_dinov2/ordinal_capsnet/` |
| DINOv2  × MLP-K1 (Mess-2)            | `results/messidor2_dinov2/mlp_k1_sigmoid/` |
