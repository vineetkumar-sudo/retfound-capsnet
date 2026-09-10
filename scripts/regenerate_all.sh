#!/usr/bin/env bash
# Rebuild every artifact that is gitignored: cached features, per-fold
# predictions, and the metric tables computed from them. Run from the repo root.
#
# Manual prerequisites (none of these can be fetched by script):
#   data/weights/RETFound_mae_natureCFP.pth   HuggingFace YukunZhou/RETFound_mae_natureCFP (gated, accept terms)
#   APTOS competition rules accepted at kaggle.com/competitions/aptos2019-blindness-detection/rules
#   data/messidor2/IMAGES/                    ADCIS registration (adcis.net)
#   data/idrid/"B. Disease Grading"/          ieee-dataport registration
#
# Total runtime is dominated by the LoRA fine-tune (~16 h on MPS for 3 seeds);
# everything else is roughly an hour. Run under nohup and check the log.
#
# ponytail: straight-line script, no stage selection. Re-run it after deleting
# only the stage you want rebuilt -- every step below is idempotent or wipes
# its own output directory first.

set -euo pipefail

SEEDS=42,123,456

echo "=== APTOS: download + unpack ==="
if [ ! -f data/aptos/train.csv ]; then
    uv run kaggle competitions download -c aptos2019-blindness-detection -p data/aptos/
    unzip -q -o data/aptos/aptos2019-blindness-detection.zip -d data/aptos/
fi

echo "=== Messidor-2 labels (images must already be in data/messidor2/IMAGES) ==="
if [ ! -f data/messidor2/messidor_data.csv ]; then
    uv run kaggle datasets download -d google-brain/messidor2-dr-grades -p data/messidor2/ --unzip
fi

echo "=== Feature caches ==="
uv run python src/data/feature_cache.py                    # APTOS x RETFound
uv run python scripts/extract_dinov2_features.py           # APTOS + Messidor-2 x DINOv2
uv run python scripts/extract_messidor2_features.py        # Messidor-2 x RETFound
uv run python scripts/extract_idrid_features.py            # IDRiD x RETFound

echo "=== APTOS: frozen head x backbone grid (3 seeds x 5 folds each) ==="
uv run python scripts/run_ordinal_capsnet.py --config configs/ordinal_capsnet.yaml        --seeds $SEEDS --no-wandb
uv run python scripts/run_ordinal_capsnet.py --config configs/mlp_ordinal.yaml            --seeds $SEEDS --no-wandb
uv run python scripts/run_ordinal_capsnet.py --config configs/ordinal_capsnet_dinov2.yaml --seeds $SEEDS --no-wandb
uv run python scripts/run_ordinal_capsnet.py --config configs/mlp_ordinal_dinov2.yaml     --seeds $SEEDS --no-wandb

echo "=== Messidor-2: within-dataset grid ==="
uv run python scripts/run_messidor2.py --models ordinal_capsnet mlp_k1_sigmoid
uv run python scripts/run_messidor2.py --models ordinal_capsnet mlp_k1_sigmoid \
    --features-dir data/messidor2/features_dinov2 --output-dir results/messidor2_dinov2

echo "=== IDRiD + cross-dataset transfer ==="
uv run python scripts/run_idrid.py
uv run python scripts/cross_dataset_aptos_to_idrid.py
uv run python scripts/cross_dataset_aptos_to_messidor2.py
uv run python scripts/posthoc_rank_calibration_messidor2.py

echo "=== LoRA fine-tune (~16 h; --resume makes an interrupted run restartable) ==="
uv run python scripts/run_lora_ordinal.py --seeds $SEEDS --resume

echo "=== Metric tables ==="
uv run python scripts/compute_calibration_metrics.py
uv run python scripts/compute_extended_calibration.py
uv run python scripts/compute_conformal_metrics.py
uv run python scripts/compute_ordinal_conformal.py      # contiguous conformal + decoder ablation
uv run python scripts/compute_clinical_metrics.py       # referable / vision-threatening DR
uv run python scripts/compute_significance_tests.py

echo "=== Figures ==="
uv run python scripts/make_day6_figures.py --strict
uv run python scripts/make_day9_figures.py --strict

echo "=== Done. ==="
