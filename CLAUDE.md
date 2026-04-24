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
- **APTOS LoRA fine-tune**: RETFound ViT-L with LoRA rank 8 (α=16, dropout 0.05, targets fused `attn.qkv` in all 24 blocks) + same Ordinal CapsNet head. 3 seeds × 5-fold CV (seeds 42, 123, 456) → **QWK 0.9127 ± 0.0008** (across-seed std), Accuracy 0.8237 ± 0.0068, Macro F1 0.6627 ± 0.0105, MAE 0.214 ± 0.006. **+0.020 QWK / −0.040 MAE over frozen Ordinal CapsNet**. Per-seed means 0.9139 / 0.9123 / 0.9121 — extraordinarily tight (spread of only 0.002 QWK across seeds; across-seed std ~15× tighter than fold-level std, indicating LoRA is highly stable). ~1.08 M trainable (303 M frozen). Per-fold runtime ≈ 65–85 min on MPS (~16 h wall-clock for 3-seed 5-fold). Driver: `scripts/run_lora_ordinal.py --seeds 42,123,456 --resume` (resume skips any fold whose `preds_fold{N}.npz` already exists, so interrupted runs are safe to re-invoke). Implication for the paper: offer frozen as the compute-efficient tier (runs on consumer hardware in 2 min) and LoRA as the performance tier (1.1 M params, still pocket-size vs 307 M full fine-tune).
- **IDRiD (Day 8, 413 train / 103 test)**: Ordinal wins *val* QWK (0.7557) but MLP+MSE wins *test* QWK (0.4652 vs Ordinal 0.4456). Every model drops ~0.30 QWK val→test equally — an IDRiD split characteristic, not a CapsNet weakness.
- **Messidor-2 (Day 9, 5-fold CV on 1,744 gradable)**: Ordinal wins QWK cleanly — 0.6065 ± 0.036 vs next-best 0.5566 (MLP+MSE). Largest within-dataset lead in the paper. Per-class: Mild +28.5 pp, Severe +24.0 pp, PDR +20.0 pp vs Vanilla (Grade 0 regresses −18 pp — the known safety-bias tradeoff).

### Cross-dataset (frozen backbone → unseen hospital/camera)
- **APTOS → IDRiD** (Day 8): QWK 0.273 (Ordinal ensemble). All models drop heavily; IDRiD cameras differ from Aravind imagers.
- **APTOS → Messidor-2** (Day 9): QWK collapses to ~0.01 for every model. **Diagnosis: the ordinal heads' mean P(y>0) drops from 0.51 on APTOS to 0.087 on Messidor-2**, so every head fires "No" on virtually every sample and predictions collapse to Grade 0 for 1741/1744 images. Binary AUC best case 0.611 (Ordinal). This is a domain-shift finding, not a method bug — rank calibration (match predicted positive rate to train rate, as used in the earlier Kaggle submission) would likely recover a substantial fraction, but was not applied here per the "no silent fallbacks" rule.

### Loss ablations (APTOS)
- **Day 4** (asymmetric ordinal loss): λ=1.0/1.0 A_baseline essentially matches vanilla ordinal — 3-seed QWK 0.8923 ± 0.0014 vs Ordinal 0.8932 ± 0.0004 (−0.0009, inside combined noise). Previous runs reported bit-identical 0.8932 = 0.8932 via a shared code path; after the Day-11 clean rerun the asymmetric loss constructor routes through a slightly different RNG path, producing a tiny seed-level drift. K-1 decomposition already produces safety bias (U/O < 1) — direction weighting is redundant.
- **Day 5A** (KC Loss / differentiable QWK at γ=0.3): slightly worse across every column (QWK 0.8914 vs 0.8932, MAE 0.264 vs 0.254). Negative result.

### Architecture ablations (APTOS)
- **Day 11** (Gogulamudi non-uniform squash, `v = s / (1 + ‖s‖)` replacing Sabour's `v = (‖s‖² / (1 + ‖s‖²))·(s/‖s‖)` inside both PrimaryCaps and all four DigitCaps routing iterations, everything else held at the Day-3 champion config): 3 seeds × 5-fold CV (42, 123, 456) → QWK 0.8894 ± 0.0006, Acc 0.7857 ± 0.0059, F1 0.6097 ± 0.0095, MAE 0.264 ± 0.004. Vs. frozen Ordinal CapsNet 3-seed baseline (QWK 0.8932 ± 0.0004) this is −0.0038 QWK, −0.007 Acc, −0.016 F1, +0.010 MAE — small but genuinely outside seed noise (both stds are < 0.001). Negative result confirmed at multi-seed granularity. Driver: `scripts/run_ordinal_capsnet.py --config configs/ordinal_capsnet_nonuniform.yaml --seeds 42,123,456 --no-wandb` (~10 min, MPS). Plumbed via `squash_variant` kwarg on PrimaryCaps / DigitCaps / OrdinalCapsNet (default `"sabour"` keeps every prior run bit-identical).
- **Day 12** (MLP + K-1 sigmoid ordinal head — direct no-capsule comparator): 3 seeds × 5-fold CV (42, 123, 456), identical K-1 decomposition + OrdinalMarginLoss + cached RETFound features, only the capsule machinery swapped for MLP trunk (1024 → 256 → 256 → 4 sigmoid heads, 329K trainable — capacity-matched to OrdinalCapsNet's 295K). Result: QWK **0.8866 ± 0.0004**, Acc **0.7963 ± 0.0041**, F1 **0.6250 ± 0.0048**, MAE **0.257 ± 0.003**, holdout QWK 0.8742 ± 0.0043. Isolates two contributions: **(a) K-1 decomposition alone** (MLP+CE 0.8783 → MLP+K1 0.8866) buys **+0.0083 QWK** and **−0.002 MAE**; **(b) capsule routing on top of K-1** (MLP+K1 0.8866 → Ordinal CapsNet 0.8932) adds a further **+0.0066 QWK** and **−0.003 MAE**. Both contributions are outside seed noise (all three rows have std ≤ 0.0004 across seeds). Capsules therefore explain ~45% of the total ordinal-head lift over plain MLP+CE. Implication for the paper: move the narrative weight onto UQ (capsule length is a calibrated probability, MLP sigmoid is not natively so) rather than onto raw QWK, where capsule contribution is small-but-clean. Driver: `scripts/run_ordinal_capsnet.py --config configs/mlp_ordinal.yaml --seeds 42,123,456 --no-wandb` (~6 min MPS), ~0.29s/epoch. Plumbed via `model.arch: mlp` dispatch in `build_model`; default `capsnet` keeps all prior runs bit-identical.

### Backbone ablations

- **Day 12B** (DINOv2 ViT-L/14 backbone ablation on APTOS + Messidor-2): frozen `dinov2_vitl14` via torch.hub, 1024-dim CLS (matches RETFound's feature dim), identical Resize-256 bicubic + CenterCrop-224 + ImageNet-norm preprocessing, same K-1 OrdinalMarginLoss head. Four cells, 3 seeds × 5-fold each. **APTOS**: Ordinal CapsNet **0.9089 ± 0.0021** (+0.0157 vs RETFound), MLP+K1 **0.9034 ± 0.0002** (+0.0168 vs RETFound). **Messidor-2 within**: Ordinal CapsNet **0.7525 ± 0.0222** (+0.146 QWK vs RETFound 0.6065), MLP+K1 **0.7643 ± 0.0148** (+0.179 QWK vs RETFound 0.5852). **Key findings: (i)** DINOv2 beats RETFound at frozen-feature DR grading cleanly across both datasets and both heads, reproducing the external head-to-head literature; **(ii)** backbone effect (+0.016 APTOS / +0.146 Messidor-2) dominates the head effect (≤ +0.021 QWK) by ~10× on Messidor-2; **(iii)** on Messidor-2 DINOv2 the head ranking flips — MLP+K1 **beats** Ordinal CapsNet by +0.012 QWK, suggesting capsules' contribution saturates and can reverse under sufficiently strong features. Complete numbers + artifact paths in `results/head_backbone_grid.md`. Drivers: `scripts/extract_dinov2_features.py` (~7 min feature cache, MPS); `scripts/run_ordinal_capsnet.py --config configs/{ordinal_capsnet,mlp_ordinal}_dinov2.yaml --seeds 42,123,456`; `scripts/run_messidor2.py --features-dir data/messidor2/features_dinov2 --output-dir results/messidor2_dinov2 --models {ordinal_capsnet,mlp_k1_sigmoid}` (each ~4–6 min MPS). `run_messidor2.py` extended with `--features-dir` / `--output-dir` CLI overrides + `mlp_k1_sigmoid` model id (routes through same ordinal-loss / predict paths as `ordinal_capsnet` since MLPOrdinal emits the same `head_lengths` schema).

### Calibration + selective prediction

- **Day 13A** (ECE + MCE + AURC + risk-coverage on all 9 head×backbone configs): across APTOS (pooled 3 seeds × 5-fold, N=9,885) and Messidor-2 (5-fold, N=1,744). ECE uses 10 equal-mass bins; confidence = P(y=predicted_grade) from the chain-rule 5-class probability derived from K-1 heads. Headline result: **capsule heads are consistently WORSE calibrated than the MLP-K1 sigmoid baseline**. APTOS ECE: RETFound CapsNet 0.1514 vs RETFound MLP-K1 **0.0912** (MLP-K1 is 40% better calibrated at comparable accuracy); DINOv2 0.1151 vs 0.1015 same direction. Prediction margin and max-prob are near-tied as selectors (AURC agrees within 0.002–0.004 everywhere), so **the capsule margin is a valid selective-prediction signal but not a uniquely strong one** — a vanilla MLP-K1 sigmoid matches it. LoRA lifts APTOS accuracy to 0.8237 but does NOT improve calibration (ECE 0.1415 vs frozen 0.1514 — within 1 pt), a known fine-tuning pathology. On Messidor-2 DINOv2 × MLP-K1 is the clean Pareto winner: best Acc (0.7099), best ECE (0.0902), best AURC (0.1736). **Implication for the paper: the "capsule-native UQ is a free strong signal" claim does not survive honest ECE analysis.** Drivers: `src/calibration.py`, `scripts/compute_calibration_metrics.py` (~3 s CPU, no training). Full numbers and rebrand implications in `results/uq_findings_summary.md`.

- **Day 13B** (Split conformal prediction, LAC + APS scores, α ∈ {0.10, 0.05}): 5 random cal/test 50:50 splits per config. **Marginal coverage holds everywhere** — every row within ±0.006 of the 90% target at α=0.10 and ±0.01 at α=0.05, confirming conformal works regardless of the head's miscalibration. Mean set size at 90% coverage: APTOS **1.27–1.38** (LoRA smallest → RETFound CapsNet largest; capsule vs MLP-K1 virtually tied), Messidor-2 **1.75–2.32** (DINOv2 × MLP-K1 smallest → RETFound × CapsNet largest). However **class-conditional coverage breaks**: on Messidor-2, worst-class coverage under APS drops to **0.583** (DINOv2 × MLP-K1, C4/PDR) and **0.667** (DINOv2 × CapsNet, C2/Moderate) — 24 to 32 pp below the 0.90 marginal target. This is a known limitation of LAC/APS that motivates the 2024 NeurIPS "Augmented Label Rank Calibration" extension cited as future work. Drivers: `src/conformal.py` (LAC + APS scores + randomised tie-break), `scripts/compute_conformal_metrics.py` (~1 s CPU). Artifacts: `results/conformal_table.{md,csv}`, `results/conformal_metrics.json`.

### Methodology
- **All APTOS ablation-table rows use 3 seeds × 5-fold CV (42, 123, 456)** except LoRA (row 9) which remains single-seed due to ~10-h compute cost for multi-seed. Data splits (10 % holdout + 5-fold) are fixed by `cfg.data.seed`; only model init RNG varies. Multi-seed runners: `scripts/run_baselines.py --seeds ...`, `scripts/run_ordinal_capsnet.py --seeds ...`, or `scripts/run_capsnet.py --model-seed ... --output-dir ...` in a shell loop. Single-seed invocation preserves the legacy flat output layout for backwards-compat. Multi-seed invocation writes per-seed subdirs `plots_dir/seed{S}/`, which the Day-6 aggregator pools across.

### UQ (Day 5B, validated on 3 datasets)
- **Prediction margin** (1 − [top1 − top2] via chain-rule probs): strongest signal everywhere. APTOS p<1e-300, IDRiD p=4.7e-4, Messidor-2 within p<1e-300 & cross p=1.9e-29.
- **DigitCap entropy**: modest lift at low coverage; not a competitor to prediction margin.
- **Routing-agreement variance**: *inverted* — rejecting high-"uncertainty" samples LOWERS accuracy on APTOS. Honest negative, included in Figure D as the third curve.

### Known negative results (paper findings, not failures)
- Day 4 asymmetric loss (no gain over vanilla ordinal)
- Day 5A KC Loss (small loss vs margin loss)
- Day 11 non-uniform squash (small loss vs Sabour squash; Gogulamudi's reported gain doesn't reproduce on frozen-RETFound + K-1 binary heads)
- Cross-dataset generalisation without calibration (frozen features don't transfer out-of-domain)
- Routing-variance UQ signal is weakly inverted
