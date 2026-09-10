#!/usr/bin/env bash
# Rebuild every gitignored artifact: cached features, per-fold predictions, and
# the metric tables the paper is written from. Run from the repo root.
#
# CANONICAL HARDWARE: one NVIDIA A100-40GB. Every number in the submitted
# tables was produced on that device, and the tuning tiers (LoRA, progressive,
# full fine-tune) are impractical elsewhere -- the same LoRA run that takes
# ~26 min per 5-fold on an A100 takes ~6 h on an Apple MPS backend.
# The frozen-head tier is the exception and runs anywhere in ~2 min per 5-fold;
# §V-H reports that deliberately, since the frozen tier is bound by Python
# dispatch rather than FLOPs and gains nothing from a server GPU.
#
# Manual prerequisites (none can be fetched by script):
#   data/weights/RETFound_mae_natureCFP.pth   HuggingFace YukunZhou/RETFound_mae_natureCFP (gated)
#   APTOS rules accepted at kaggle.com/competitions/aptos2019-blindness-detection/rules
#   data/compressed/Messidor/IMAGES.zip.00{1..4}   ADCIS registration (adcis.net)
#   data/compressed/idrid/"B. Disease Grading.zip" ieee-dataport registration
#
# Every step is idempotent or wipes its own output directory first, so the
# script can be re-run after deleting only the stage you want rebuilt.

set -euo pipefail

SEEDS=42,123,456

echo "=== Datasets ==="
if [ ! -f data/aptos/train.csv ]; then
    uv run kaggle competitions download -c aptos2019-blindness-detection -p data/aptos/
    unzip -q -o data/aptos/aptos2019-blindness-detection.zip -d data/aptos/
fi
if [ ! -d data/messidor2/IMAGES ]; then
    cat data/compressed/Messidor/IMAGES.zip.00* > data/compressed/Messidor/IMAGES_joined.zip
    unzip -q -o data/compressed/Messidor/IMAGES_joined.zip -d data/messidor2/   # 1,748 images, mixed .png/.JPG
fi
[ -f data/messidor2/messidor_data.csv ] || \
    uv run kaggle datasets download -d google-brain/messidor2-dr-grades -p data/messidor2/ --unzip
[ -d "data/idrid/B. Disease Grading" ] || \
    unzip -q -o "data/compressed/idrid/B. Disease Grading.zip" -d data/idrid/

echo "=== Resized image caches (source for the 448 px ablation) ==="
uv run python scripts/build_image_cache.py --src data/aptos/train_images --dst data/aptos/train_images_512 --short-side 512
uv run python scripts/build_image_cache.py --src data/aptos/train_images --dst data/aptos/train_images_256 --short-side 256

echo "=== Feature caches ==="
uv run python src/data/feature_cache.py                     # APTOS x RETFound @224
uv run python scripts/extract_dinov2_features.py            # APTOS + Messidor-2 x DINOv2
uv run python scripts/extract_messidor2_features.py         # Messidor-2 x RETFound
uv run python scripts/extract_idrid_features.py --backbone retfound --out-dir data/idrid/features
uv run python scripts/extract_idrid_features.py --backbone dinov2   --out-dir data/idrid/features_dinov2
uv run python scripts/extract_features_multires.py --img-size 448 \
    --image-dir data/aptos/train_images_512 --out-dir data/aptos/features_448

echo "=== Frozen head x backbone grid (3 seeds x 5 folds each) ==="
for cfg in ordinal_capsnet mlp_ordinal ordinal_capsnet_dinov2 mlp_ordinal_dinov2 \
           ordinal_capsnet_448 mlp_ordinal_448; do
    uv run python scripts/run_ordinal_capsnet.py --config configs/$cfg.yaml --seeds $SEEDS --no-wandb
done
uv run python scripts/run_baselines.py --seeds $SEEDS
uv run python scripts/run_capsnet.py --model-seed 42
uv run python scripts/run_asymmetric_ordinal.py --seeds $SEEDS
uv run python scripts/run_ordinal_kc.py --seeds $SEEDS
uv run python scripts/run_ordinal_capsnet.py --config configs/ordinal_capsnet_nonuniform.yaml --seeds $SEEDS --no-wandb

echo "=== Messidor-2 and IDRiD, both backbones ==="
uv run python scripts/run_messidor2.py --models ordinal_capsnet mlp_k1_sigmoid
uv run python scripts/run_messidor2.py --models ordinal_capsnet mlp_k1_sigmoid \
    --features-dir data/messidor2/features_dinov2 --output-dir results/messidor2_dinov2
uv run python scripts/run_idrid.py
uv run python scripts/run_idrid.py --features-dir data/idrid/features_dinov2 --output-dir results/idrid_dinov2

echo "=== Cross-dataset transfer + prior-shift correction ==="
uv run python scripts/cross_dataset_aptos_to_idrid.py \
    --aptos-features data/aptos/features --idrid-features data/idrid/features \
    --out-dir results/cross_dataset/aptos_to_idrid
uv run python scripts/cross_dataset_aptos_to_idrid.py \
    --aptos-features data/aptos/features_dinov2 --idrid-features data/idrid/features_dinov2 \
    --out-dir results/cross_dataset/aptos_to_idrid_dinov2
uv run python scripts/cross_dataset_aptos_to_messidor2.py
uv run python scripts/posthoc_rank_calibration_messidor2.py

echo "=== Tuning tiers (A100; ~26 min per 5-fold for LoRA/progressive, ~54 min for full) ==="
uv run python scripts/run_lora_ordinal.py --backbone retfound --tune-mode lora        --seeds $SEEDS --resume --out-dir results/lora_ordinal_capsnet
uv run python scripts/run_lora_ordinal.py --backbone retfound --tune-mode progressive --seeds $SEEDS --resume --out-dir results/progressive_unfreeze_ordinal
uv run python scripts/run_lora_ordinal.py --backbone retfound --tune-mode full        --seeds $SEEDS --resume --out-dir results/full_finetune_ordinal
uv run python scripts/run_lora_ordinal.py --backbone dinov2   --tune-mode lora        --seeds $SEEDS --resume --out-dir results/lora_dinov2_ordinal
uv run python scripts/run_lora_ordinal.py --backbone dinov2   --tune-mode progressive --seeds 42     --resume --out-dir results/dinov2_progressive
uv run python scripts/run_lora_ordinal.py --backbone dinov2   --tune-mode full        --seeds 42     --resume --out-dir results/dinov2_full_finetune
uv run python scripts/run_lora_messidor2.py --seeds $SEEDS --out-dir results/lora_messidor2

echo "=== Metric tables ==="
uv run python scripts/compute_calibration_metrics.py     # Table: calibration + selective prediction
uv run python scripts/compute_extended_calibration.py    # Table: temperature scaling
uv run python scripts/compute_conformal_metrics.py       # Table: split conformal + Mondrian CCP
uv run python scripts/compute_ordinal_conformal.py       # Table: contiguous ordinal conformal (OCP)
uv run python scripts/compute_monotone_decode.py         # Table: rank-monotonicity projection
uv run python scripts/compute_clinical_metrics.py        # Table: referable / vision-threatening DR
uv run python scripts/compute_loss_calibration.py --seeds $SEEDS   # Table: loss x head factorial
uv run python scripts/compute_uda_baselines.py           # Table: UDA baselines
uv run python scripts/compute_significance_tests.py      # paired Wilcoxon per family
uv run python scripts/profile_compute.py                 # Table: FLOPs / peak memory / throughput

echo "=== Aggregates and figures ==="
uv run python scripts/aggregate_day6.py
uv run python scripts/aggregate_day8.py
uv run python scripts/aggregate_day9.py
uv run python scripts/make_day6_figures.py --strict
uv run python scripts/make_day9_figures.py --strict
uv run python scripts/make_lora_figures.py

echo "=== Done. Rebuild the paper with: cd paper/submission && pdflatex article.tex (x3) ==="
