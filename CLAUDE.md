# RETFound-CapsNet

## Project
- Deep learning research project: RETFound + Ordinal Regression Capsule Network for diabetic retinopathy grading
- Python 3.12, managed with `uv`
- Backbone: frozen RETFound (ViT-L) → pre-computed 1024-dim features → CapsNet head

## Structure
- `src/` — main package (models, losses, data submodules)
- `src/models/retfound_capsnet.py` — RETFound → PrimaryCaps → OrdinalDigitCaps
- `src/losses/` — margin, ordinal (K-1 BCE), asymmetric, KC loss
- `src/data/feature_cache.py` — RETFound forward pass → cached .npy files
- `src/uncertainty.py` — routing variance + DigitCap entropy UQ
- `configs/` — YAML hyperparameter configs
- `data/` — datasets, features cache, model weights (all gitignored)
- `experiments/` — run logs (gitignored)
- `results/` — final figures/tables (gitignored)

## Datasets
- APTOS 2019: `kaggle competitions download -c aptos2019-blindness-detection -p data/aptos/`
- EyePACS: `kaggle competitions download -c diabetic-retinopathy-detection -p data/eyepacs/`
- Messidor-2: register at adcis.net
- IDRiD: register at ieee-dataport.org
- RETFound weights: `data/weights/RETFound_cfp_weights.pth`
- Kaggle CLI uses legacy API key (~/.kaggle/kaggle.json), NOT access tokens (KGAT_*)

## Commands
- `uv run python main.py` — run entrypoint
- `uv add <pkg>` — add dependency
