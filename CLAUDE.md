# RETFound-CapsNet

## Project
- Deep learning research project: RETFound + Ordinal Regression Capsule Network for diabetic retinopathy grading on APTOS 2019.
- Python 3.12, managed with `uv`.
- Architecture: frozen RETFound (ViT-L, nature-CFP MAE) → cached 1024-dim CLS features → CapsNet head.
- Each "Day N" experiment is a self-contained script in `scripts/`; no central CLI entrypoint.

## Structure
- `src/models/`
  - `baselines.py` — linear + MLP heads (Day 1)
  - `capsnet.py` — vanilla 5-class CapsNet with dynamic routing + routing-history hook (Day 2 / Day 5 UQ)
  - `ordinal_capsnet.py` — shared PrimaryCaps + K-1 binary DigitCaps heads (Day 3 champion)
- `src/losses/`
  - `margin_loss.py` — Sabour margin loss
  - `ordinal_loss.py` — K-1 binary ordinal decomposition + `predict_grade_from_heads`
  - `asymmetric_loss.py` — direction-aware ordinal margin loss (Day 4)
  - `kc_loss.py` — differentiable QWK via chain-rule class probs (Day 5A)
- `src/data/`
  - `feature_cache.py` — idempotent RETFound forward pass → `data/aptos/features/*.npy`
  - Preprocessing follows the official RETFound eval transform: Resize 256 (bicubic) → CenterCrop 224 → ImageNet normalize.
- `src/evaluate.py` — `compute_all_metrics` (QWK, accuracy, macro-F1, MAE, confusion matrix) — single source of metrics for Day 6+ aggregation.
- `src/uncertainty.py` — digit-cap entropy, routing-agreement variance, prediction margin
- `scripts/` — one `run_*.py` / `make_*.py` / `aggregate_*.py` per experiment day
- `configs/` — YAML hyperparameters per experiment
- `data/`, `experiments/`, `results/` — all gitignored

## Datasets
- APTOS 2019: `uv run kaggle competitions download -c aptos2019-blindness-detection -p data/aptos/`
- EyePACS (pretrain — not yet used): `uv run kaggle competitions download -c diabetic-retinopathy-detection -p data/eyepacs/`
- Messidor-2: register at adcis.net
- IDRiD: register at ieee-dataport.org
- RETFound weights: `data/weights/RETFound_mae_natureCFP.pth` (ViT-L, nature CFP pretraining)
- Kaggle CLI lives in `.venv/bin/kaggle` — invoke via `uv run kaggle …`. Auth uses legacy `~/.kaggle/kaggle.json`, NOT KGAT_* access tokens.
- APTOS is a Kernels-only competition: direct CSV submission returns 400. Real submissions require a Kaggle notebook; local CV / holdout is the primary evaluator for this project.

## Evaluation protocol
- 10% of train held back as frozen holdout before CV (seed 42).
- 5-fold stratified CV on the remaining 90%.
- Primary metric: Quadratic Weighted Kappa (QWK). Also tracked: accuracy, per-class recall, `count_non_monotonic` for ordinal heads.
- Early stopping: patience 30 on val QWK. AdamW, weight_decay=1e-4.
- W&B tracking is enabled per script with per-day run groups.

## Commands
- `uv run python scripts/run_baselines.py` — Day 1 linear/MLP baselines (CE, MSE, weighted CE)
- `uv run python scripts/run_capsnet.py` — Day 2 vanilla 5-class CapsNet + margin loss
- `uv run python scripts/check_ordinal_capsnet.py` — Day 3 sanity checks (shapes, gradients, convergence, monotonicity)
- `uv run python scripts/run_ordinal_capsnet.py` — Day 3 K-1 ordinal CapsNet (champion); supports `--seeds 42,123,456`
- `uv run python scripts/sweep_asymmetric.py` / `run_asymmetric_ordinal.py` — Day 4 asymmetric loss
- `uv run python scripts/run_ordinal_kc.py` — Day 5A KC Loss
- `uv run python scripts/run_uq_analysis.py` — Day 5B standalone UQ study (legacy; UQ figures are now produced by `make_day6_figures.py` from the patched ordinal preds)
- `uv run python scripts/aggregate_day6.py` / `make_day6_figures.py [--strict]` — Day 6 ablation table + APTOS figures
- `uv run python scripts/make_architecture_diagram.py` — Day 7 architecture pipeline figure
- `uv run python scripts/extract_idrid_features.py` / `run_idrid.py` / `cross_dataset_aptos_to_idrid.py` / `aggregate_day8.py` / `make_day8_figures.py` — Day 8 IDRiD + cross-dataset
- `uv add <pkg>` — add dependency

## Experimental findings
- **Day 3 champion**: ordinal CapsNet + margin loss → val QWK 0.894, holdout QWK 0.881.
- **Day 4** (asymmetric ordinal loss): no gain — K-1 decomposition already produces safety bias (U/O < 1), so direction weighting is redundant.
- **Day 5A** (KC Loss / differentiable QWK): no gain over margin loss.
- **Day 5B** (UQ): prediction margin is by far the strongest uncertainty signal (p<1e-300; 96.3% acc at 50% coverage). Digit-cap entropy is moderately useful. Routing-agreement variance is weak / slightly inverted.
- Negative results (Day 4, 5A) are treated as paper findings, not failures.
