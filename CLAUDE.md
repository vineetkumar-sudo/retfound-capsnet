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
- `uv run python scripts/extract_messidor2_features.py` / `run_messidor2.py` / `cross_dataset_aptos_to_messidor2.py` / `aggregate_day9.py` / `make_day9_figures.py [--strict]` — Day 9 Messidor-2 + cross-dataset
- `uv run python scripts/run_lora_ordinal.py [--resume] [--force-redo]` — LoRA fine-tune: RETFound ViT-L (frozen) + LoRA r=8 adapters on `attn.qkv` + Ordinal CapsNet head. `--resume` skips folds whose `preds_fold{N}.npz` already exists — safe to Ctrl-C then restart. No W&B by design.
- `uv add <pkg>` — add dependency

## Experimental findings

### Within-dataset
- **APTOS Day 3 champion (frozen backbone)**: ordinal CapsNet + margin loss → 3 seeds × 5-fold CV QWK 0.8932 ± 0.0004 (MAE 0.254 ± 0.002 — best in ablation table), frozen 10% holdout QWK ≈ 0.883. Wins QWK + MAE; MLP+CE wins Accuracy narrowly (0.8064), Vanilla CapsNet wins Macro F1 (0.6327) — ordinal trades raw top-1 accuracy for fewer distant errors (what QWK / MAE reward). ~295K trainable params, 115 s full 5-fold CV on MPS.
- **APTOS LoRA fine-tune (Day 10-ish exploratory)**: RETFound ViT-L with LoRA rank 8 (α=16, dropout 0.05, targets fused `attn.qkv` in all 24 blocks) + same Ordinal CapsNet head. Single-seed 5-fold CV → QWK 0.9139 ± 0.0127 (every fold individually beats frozen), Accuracy 0.8206, Macro F1 0.6576, MAE 0.215. **+0.021 QWK / −0.039 MAE over frozen**. ~1.08 M trainable (303 M frozen). Per-fold runtime ≈ 65–85 min on MPS. Driver: `scripts/run_lora_ordinal.py --epochs 20 --patience 8 --batch-size 8 --resume`. Implication for the paper: offer frozen as the compute-efficient tier (runs on consumer hardware in 2 min) and LoRA as the performance tier (1.1 M params, still pocket-size vs 307 M full fine-tune).
- **IDRiD (Day 8, 413 train / 103 test)**: Ordinal wins *val* QWK (0.7557) but MLP+MSE wins *test* QWK (0.4652 vs Ordinal 0.4456). Every model drops ~0.30 QWK val→test equally — an IDRiD split characteristic, not a CapsNet weakness.
- **Messidor-2 (Day 9, 5-fold CV on 1,744 gradable)**: Ordinal wins QWK cleanly — 0.6065 ± 0.036 vs next-best 0.5566 (MLP+MSE). Largest within-dataset lead in the paper. Per-class: Mild +28.5 pp, Severe +24.0 pp, PDR +20.0 pp vs Vanilla (Grade 0 regresses −18 pp — the known safety-bias tradeoff).

### Cross-dataset (frozen backbone → unseen hospital/camera)
- **APTOS → IDRiD** (Day 8): QWK 0.273 (Ordinal ensemble). All models drop heavily; IDRiD cameras differ from Aravind imagers.
- **APTOS → Messidor-2** (Day 9): QWK collapses to ~0.01 for every model. **Diagnosis: the ordinal heads' mean P(y>0) drops from 0.51 on APTOS to 0.087 on Messidor-2**, so every head fires "No" on virtually every sample and predictions collapse to Grade 0 for 1741/1744 images. Binary AUC best case 0.611 (Ordinal). This is a domain-shift finding, not a method bug — rank calibration (match predicted positive rate to train rate, as used in the earlier Kaggle submission) would likely recover a substantial fraction, but was not applied here per the "no silent fallbacks" rule.

### Loss ablations (APTOS)
- **Day 4** (asymmetric ordinal loss): bit-identical to vanilla ordinal when symmetric λ=1.0/1.0 used (A_baseline). K-1 decomposition already produces safety bias (U/O < 1) — direction weighting is redundant.
- **Day 5A** (KC Loss / differentiable QWK at γ=0.3): slightly worse across every column (QWK 0.8914 vs 0.8932, MAE 0.264 vs 0.254). Negative result.

### UQ (Day 5B, validated on 3 datasets)
- **Prediction margin** (1 − [top1 − top2] via chain-rule probs): strongest signal everywhere. APTOS p<1e-300, IDRiD p=4.7e-4, Messidor-2 within p<1e-300 & cross p=1.9e-29.
- **DigitCap entropy**: modest lift at low coverage; not a competitor to prediction margin.
- **Routing-agreement variance**: *inverted* — rejecting high-"uncertainty" samples LOWERS accuracy on APTOS. Honest negative, included in Figure D as the third curve.

### Known negative results (paper findings, not failures)
- Day 4 asymmetric loss (no gain over vanilla ordinal)
- Day 5A KC Loss (small loss vs margin loss)
- Cross-dataset generalisation without calibration (frozen features don't transfer out-of-domain)
- Routing-variance UQ signal is weakly inverted
