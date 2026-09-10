# Head × backbone grid — artifact map

> **This file no longer carries numbers.** It used to hold a hand-maintained copy
> of the 2×2 grid, and that copy went stale when the experiments were re-run on
> the A100. The authoritative grid is **Table II (`tab:head_backbone`) of
> `paper/submission/article.tex`**, whose cells were audited one by one against
> the per-fold predictions, backed by `results/aptos_final_table.md` and
> `results/messidor2_final_table.md`. What this file keeps is the part that does
> not go stale: which directory holds which cell.

## Protocol

All cells share preprocessing, the K−1 decomposition, the loss, and the splits.
The only variables are the **backbone** — frozen RETFound ViT-L/16 nature-CFP MAE
vs frozen DINOv2 ViT-L/14 (`dinov2_vitl14`) — and the **head** — Ordinal CapsNet
with dynamic routing vs MLP with 4 independent sigmoid ordinal heads. Both
backbones emit 1024-dim CLS features at 224 px under an identical Resize-256
bicubic → CenterCrop-224 → ImageNet-normalise pipeline, so the backbone weights
are the only thing that changes. APTOS is 3 seeds × 5-fold CV; Messidor-2 is
5-fold CV on the 1,744 gradable images.

## Which directory is which cell

| Cell | Path |
|---|---|
| RETFound × Ordinal CapsNet (APTOS) | `results/ordinal_capsnet/seed{42,123,456}/` |
| RETFound × MLP-K1 (APTOS) | `results/mlp_ordinal/seed{42,123,456}/` |
| DINOv2 × Ordinal CapsNet (APTOS) | `results/ordinal_capsnet_dinov2/seed{42,123,456}/` |
| DINOv2 × MLP-K1 (APTOS) | `results/mlp_ordinal_dinov2/seed{42,123,456}/` |
| RETFound × Ordinal CapsNet + LoRA (APTOS) | `results/lora_ordinal_capsnet/` |
| RETFound × Ordinal CapsNet (Messidor-2) | `results/messidor2/ordinal_capsnet/` |
| RETFound × MLP-K1 (Messidor-2) | `results/messidor2/mlp_k1_sigmoid/` |
| DINOv2 × Ordinal CapsNet (Messidor-2) | `results/messidor2_dinov2/ordinal_capsnet/` |
| DINOv2 × MLP-K1 (Messidor-2) | `results/messidor2_dinov2/mlp_k1_sigmoid/` |
| RETFound / DINOv2 × all heads (IDRiD) | `results/idrid/`, `results/idrid_dinov2/` |

Each directory holds `preds_fold{N}.npz` (`y_true`, `y_pred`, `head_probs` or
`head_lengths`), `summary.json`, and the per-run plots. The `.npz` files are
gitignored; `scripts/regenerate_all.sh` rebuilds them.

## Commands

```bash
# Feature extraction (DINOv2 ViT-L/14 via torch.hub)
uv run python scripts/extract_dinov2_features.py

# APTOS, 3 seeds x 5-fold each
uv run python scripts/run_ordinal_capsnet.py --config configs/ordinal_capsnet_dinov2.yaml --seeds 42,123,456 --no-wandb
uv run python scripts/run_ordinal_capsnet.py --config configs/mlp_ordinal_dinov2.yaml      --seeds 42,123,456 --no-wandb

# Messidor-2 within-dataset, 5-fold
uv run python scripts/run_messidor2.py --features-dir data/messidor2/features_dinov2 --output-dir results/messidor2_dinov2 --models ordinal_capsnet
uv run python scripts/run_messidor2.py --features-dir data/messidor2/features_dinov2 --output-dir results/messidor2_dinov2 --models mlp_k1_sigmoid
```

## The findings, without numbers

1. **The backbone effect dominates the head effect** on both datasets — by
   roughly an order of magnitude on Messidor-2.
2. **DINOv2 beats RETFound at the same parameter count**, on both datasets and
   for both heads, reproducing the external head-to-head literature: a generalist
   self-supervised ViT beats the domain-specific RETFound at frozen-feature DR
   grading.
3. **Messidor-2 flips the head ranking under the stronger backbone.** On
   RETFound features the capsule head wins; on DINOv2 features the two heads
   tie. Capsule contribution saturates under sufficiently strong features.
4. **The K−1 decomposition is the more consistent architectural lever** than
   capsule routing, on an identical trunk and identical backbone.
