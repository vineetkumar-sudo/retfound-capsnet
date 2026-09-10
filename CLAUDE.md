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
- `data/` — gitignored (raw images, feature caches, backbone weights).
- `results/` — **tracked**: metric summaries, aggregated tables, figures. The
  `.npy`/`.npz` prediction and feature arrays inside it are gitignored.

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
- `uv run python scripts/run_lora_ordinal.py [--resume] [--force-redo]` — LoRA fine-tune: RETFound ViT-L (frozen) + LoRA r=8 adapters on `attn.qkv` + Ordinal CapsNet head. `--resume` is a footgun: it reads the tracked `summary.json` *before* globbing `preds_fold{N}.npz`, so on a fresh clone it skips every fold and exits in seconds while printing stale numbers. Use `--force-redo`, or delete the stale `summary.json` files first. No W&B by design.
- `bash scripts/regenerate_all.sh` — **canonical end-to-end pipeline on one A100**:
  image caches, 224/448 feature extraction for both backbones, all six
  backbone×tune-mode tiers, Messidor-2 and IDRiD arms, then all ten metric
  scripts. This is the single source of every number in the paper.
- `uv add <pkg>` — add dependency

## Experimental findings

> **Provenance — read this before quoting any number.** Every figure below comes
> from the **A100 rerun (September 2026)**, which is the canonical hardware for
> the paper. The earlier MPS-era numbers that this section used to record
> (frozen QWK 0.8932, MLP+K-1 0.8866, Messidor-2 0.6065, ECE 0.1514, LoRA
> 0.9127, and the "DINOv2 × MLP-K1 is the Pareto winner" claim) are
> **superseded**: the rerun moved several of them enough to change conclusions,
> not just digits. The authoritative sources, in order, are
> (1) the tables in `paper/submission/article.tex`, which have been audited
> cell-by-cell against `preds_fold*.npz`, (2) the script-generated tables in
> `results/*.{md,csv}`, and (3) the raw per-fold predictions. Regenerate
> everything with `scripts/regenerate_all.sh`. Read pre-rounded CSVs with care —
> deriving a table from a 4-decimal CSV double-rounds.

### APTOS-2019 (3 seeds × 5-fold CV unless noted)

| Config | QWK | Acc | Macro-F1 | MAE | Trainable |
|---|---|---|---|---|---|
| Linear probe + CE | 0.8389 ± 0.0122 | 0.7971 | 0.6011 | 0.291 | — |
| MLP + CE | 0.8755 ± 0.0014 | 0.8091 | 0.6325 | 0.258 | 132 K |
| MLP + K-1 sigmoid | 0.8849 ± 0.0015 | 0.7976 | 0.6235 | 0.258 | 329 K |
| Vanilla CapsNet | 0.8696 ± 0.0026 | 0.8061 | 0.6301 | 0.263 | — |
| **Ordinal CapsNet (frozen)** | **0.8923 ± 0.0002** | 0.7935 | 0.6259 | 0.256 | **295 K** |
| **Ordinal CapsNet + LoRA r=8** | **0.9143 ± 0.0006** | **0.8240** | **0.6701** | **0.212** | 1.08 M |

Contribution split on identical features: K-1 decomposition alone buys
**+0.0094** QWK (MLP+CE → MLP+K-1); capsule routing on top adds a further
**+0.0074** (MLP+K-1 → Ordinal CapsNet). Both outside seed noise (all stds
≤ 0.0015), but the head effect is small — the paper's weight sits on the
ordinal-coherence and UQ machinery, not on raw QWK.

### Fine-tuning tiers (APTOS, identical splits)

Frozen 0.8923 → LoRA r=8 **0.9143** → progressive (last 4 blocks) **0.9189**
→ full fine-tune **0.8880**. Full fine-tuning is *worse than training no
backbone weights at all* despite 1,031× more trainable parameters — 2,636
training images cannot support 304 M. LoRA and progressive are statistically
indistinguishable, so LoRA wins on 47× fewer parameters. The
parameter-efficiency curve is **non-monotonic**.

### Backbone: DINOv2 beats RETFound everywhere

APTOS QWK: CapsNet 0.8923 → **0.9074** (+0.0152), MLP+K-1 0.8849 → **0.9041**
(+0.0192). Messidor-2 within: CapsNet 0.6021 → **0.7586** (+0.1565), MLP+K-1
0.5900 → **0.7587** (+0.1687). The backbone effect dominates the head effect by
~10× on Messidor-2, and under DINOv2 the two heads are a dead tie there
(−0.0002) — capsule contribution saturates under stronger features.

**Input resolution is not the lever either.** Re-extracting at 448 px (position
embedding resampled 14×14 → 28×28) makes grading *worse*: −0.0094 QWK for the
capsule head, −0.0079 for MLP+K-1, consistent across all three seeds. RETFound
was pretrained at 224 and pays more for the token-grid mismatch than it gains
in lesion detail.

### Calibration — the honest-audit results

- The capsule head is **consistently worse calibrated** than MLP+K-1 at
  comparable accuracy. APTOS ECE 0.143 vs 0.098; Messidor-2 0.103 vs 0.083.
  This survived every re-run; it is the central negative result.
- **The heads are UNDER-confident, not over-confident** — in all 10
  equal-mass bins, and the fitted temperature is < 1. The submitted paper
  claimed the opposite in both the prose and a table caption. Do not
  reintroduce the over-confidence framing.
- **Loss × head factorial separates two confounded causes.** Swapping the
  Sabour margin loss for plain BCE improves ECE by −0.079 (capsule) and
  −0.087 (MLP). At matched loss the capsule head is still worse by +0.032 to
  +0.061. The original single-cause attribution to the margin loss was half
  right. Best cell: MLP+K-1 with BCE at ECE **0.0181** vs 0.1581 for
  capsule+margin — 8.7× better calibrated for −0.0065 QWK.
- LoRA raises accuracy to 0.824 but does **not** fix calibration
  (ECE 0.131 vs frozen 0.143) — the known fine-tuning pathology.
- Prediction margin and max-prob are near-tied as selective-prediction
  signals (AURC within 0.002–0.004 everywhere), so the capsule margin is a
  *valid* but not *uniquely strong* signal. The "capsule-native UQ is a free
  strong signal" claim does not survive.

### Ordinal coherence (`src/calibration.py`)

The legacy product decode `q_{k-1}(1−q_k)` requires renormalisation on
**86–100% of samples**, because the row sum is ≥ 1 by algebra except in
degenerate saturated cases. It is a property of the decode itself, *not* a
DINOv2 or a miscalibration symptom, which is what the submitted paper implied. Rank-monotonicity violations in the raw heads run 4.2%–62.2%
depending on cell. Projecting onto the antitonic cone ∩ unit box (exact, via
PAVA, O(K)) then decoding by telescoping differences roughly **halves ECE**
in every configuration: APTOS capsule 0.1469 → **0.0715**, Messidor-2 MLP+K-1
0.0860 → **0.0387**. The one exception is the already-well-calibrated
MLP+BCE cell, where the projection is marginally harmful (0.0221 vs 0.0181).
`mode="isotonic"` is the default decoder; `"difference"` and `"product"`
reproduce the legacy paths.

### Conformal prediction (`src/conformal.py`)

Marginal coverage holds everywhere (within ±0.006 of 90% at α=0.10), which is
the point — conformal works regardless of head miscalibration. Mean set size
1.28–1.38 on APTOS, 1.83–2.33 on Messidor-2. **Class-conditional coverage
breaks** without Mondrian conditioning: worst-class coverage falls as low as
0.583 under APS. LAC/APS/RAPS also emit **non-contiguous** grade sets 0.6%–12.8%
of the time, which is clinically meaningless for an ordinal scale; **OCP**
(nested-interval score) removes all of them at +0.04 mean set size and beats
the interval-hull baseline in all 8 cells.

### Cross-dataset transfer

- **APTOS → IDRiD**: RETFound QWK 0.2788 → DINOv2 **0.6259**. That is within
  noise of training on IDRiD directly (0.6573) and clears the ≈0.60
  clinical-agreement bar. The submitted paper's pessimistic transfer
  conclusion is **specific to RETFound-MAE features**, not general.
- **APTOS → Messidor-2**: collapses to QWK ~0.017 for every model — a
  *threshold* collapse, not a feature-space failure. Mean `P(y>0)` drops
  0.51 → 0.087, so every head fires "No" and 99.7% of images are predicted
  Grade 0, while binary AUC is preserved at 0.620. SLD prior correction with
  **no target labels** recovers QWK to **0.2399** and referable sensitivity
  from 2.0% to 53.8% (oracle prior upper bound 0.2783). A stronger backbone
  alone does not fix this one.
- **IDRiD within-dataset** (413 train / 103 official test): every model loses
  0.21–0.35 QWK val → test. Roughly a third of that gap was the backbone, not
  the split — DINOv2 lifts test QWK by +0.196 to +0.220 on all four models.

### Clinical operating points (referable DR, grade ≥ 2, at `P(y>1)` = 0.5)

Every APTOS row clears the NHS DESP bar (sens ≥ 0.85, spec ≥ 0.80) — best is
RETFound × Ordinal + LoRA at sens 0.954 / spec 0.931. **Every Messidor-2 row
fails**, best sens 0.744 (DINOv2 × MLP+K-1) against the 0.85 minimum. The
screening claim is APTOS-only and the paper says so.

### Methodology

- All APTOS ablation rows are 3 seeds × 5-fold CV (42, 123, 456), LoRA
  included. Splits are fixed by `cfg.data.seed`; only model-init RNG varies.
- 10% frozen holdout carved out before CV (seed 42), then 5-fold stratified
  on the remaining 90%.
- Multi-seed runs write `plots_dir/seed{S}/`; single-seed keeps the legacy
  flat layout for backwards compatibility.
- **`run_lora_ordinal.py --resume` reads the tracked `summary.json` before
  globbing predictions**, so on a fresh clone it skips every fold and exits in
  seconds while printing stale numbers. Use `--force-redo`, or delete the
  stale `summary.json` files first.

### Known negative results (paper findings, not failures)

- Asymmetric ordinal loss — no gain over vanilla ordinal (K-1 already
  produces safety bias, U/O < 1, so direction weighting is redundant).
- KC Loss / differentiable QWK at γ=0.3 — slightly worse on every column.
- Gogulamudi non-uniform squash — small loss vs Sabour squash; the reported
  gain does not reproduce on frozen RETFound + K-1 binary heads.
- Full fine-tuning — worse than a frozen backbone (see tiers above).
- 448 px input — worse than 224 px for both heads.
- Focal loss — worse calibration than both margin and BCE.
- Routing-agreement variance as a UQ signal — *inverted*: rejecting
  high-"uncertainty" samples lowers accuracy on APTOS. Reported honestly as
  the third curve in the UQ figure.
- Cross-dataset transfer without any correction (see above).
