# RETFound-CapsNet

**Trustworthy AI for Diabetic Retinopathy Grading: Auditing Foundation Models, Capsule Networks, and Conformal Uncertainty.**

This repository contains the code, configs, and lightweight result artifacts that accompany the paper in `paper/submission/article.pdf`. It runs a fully-crossed audit of the three design choices in frozen-feature DR grading — the pretrained backbone (RETFound-MAE vs DINOv2), the head (ordinal capsule vs plain MLP with K−1 sigmoid thresholds), and the uncertainty layer (capsule-native confidence vs temperature scaling vs split conformal prediction) — on APTOS-2019, IDRiD, and Messidor-2.

The short version of what we found: **the backbone is the dominant lever, the capsule head is a marginal one, and the capsule-native uncertainty signal does not survive an honest calibration audit.** Conformal prediction is the uncertainty layer that holds up, and only with a per-class quantile.

![DR grade examples](results/figures/fig_0_grade_samples.png)

## Abstract

Automated diabetic retinopathy (DR) grading needs both accuracy and trustworthy uncertainty, but it is unclear which design choices actually drive either. The three candidates are the pretrained backbone, the prediction head on top, and the uncertainty layer wrapped around it. We run a fully-crossed ablation that varies all three on APTOS-2019 and Messidor-2, comparing a domain-specific backbone (RETFound-MAE) against a generalist one (DINOv2), a capsule-network head against a plain MLP with K−1 ordinal sigmoid thresholds, and capsule-native confidence against temperature scaling and split conformal prediction. Features are extracted once and frozen, so only the head trains, with three seeds and five-fold cross-validation. The backbone is the dominant lever. Swapping RETFound-MAE for DINOv2 lifts QWK by an order of magnitude more than swapping the head, while capsule routing adds little over the much simpler MLP+K−1 baseline and produces a systematically *under*-confident signal that a single-parameter temperature rescaling largely fixes. Split conformal prediction is the uncertainty layer that holds up. It delivers the requested coverage on average, but coverage collapses on rare grades unless a per-class quantile is used. Out of distribution, transfer from APTOS to Messidor-2 collapses to a single predicted grade until a prior-shift correction is applied, after which performance recovers part-way but stays well below clinical agreement. The contribution is a calibrated map of what helps frozen-feature DR grading, what does not, and where the in-domain numbers stop being trustworthy.

## Architecture

![Architecture](results/figures/fig_architecture.png)

RETFound (frozen or LoRA-adapted on `attn.qkv`) → PrimaryCaps → four K−1 binary DigitCaps heads with dynamic routing → chain-rule 5-class probabilities + prediction margin UQ.

## Headline results (APTOS-2019, 3 seeds × 5-fold CV)

All numbers below come from the A100 rerun that is canonical for the paper. Regenerate
everything with `bash scripts/regenerate_all.sh`.

| Model | QWK | Accuracy | Macro-F1 | MAE | Trainable params |
|---|---:|---:|---:|---:|---:|
| Linear + CE | 0.8389 ± 0.0122 | 0.7971 | 0.6011 | 0.291 | 5K |
| MLP + CE | 0.8755 ± 0.0014 | 0.8091 | 0.6325 | 0.258 | 132K |
| MLP + K−1 sigmoid | 0.8849 ± 0.0015 | 0.7976 | 0.6235 | 0.258 | 329K |
| Vanilla 5-class CapsNet | 0.8696 ± 0.0026 | 0.8061 | 0.6301 | 0.263 | ~300K |
| **Ordinal CapsNet (frozen)** | **0.8923 ± 0.0002** | 0.7935 | 0.6259 | 0.256 | **295K** |
| **Ordinal CapsNet + LoRA r=8** | **0.9143 ± 0.0006** | **0.8240** | **0.6701** | **0.212** | 1.08M |

The head is the smaller lever. Swapping the **backbone** to frozen DINOv2 ViT-L/14 beats
all of it: APTOS QWK 0.8923 → **0.9074** for the capsule head and 0.8849 → **0.9041** for
MLP+K−1, and on Messidor-2 the same swap is worth **+0.157 QWK**, roughly ten times the
head effect. Under DINOv2 the two heads tie on Messidor-2 (−0.0002).

Fine-tuning is non-monotonic in trainable parameters: frozen 0.8923 → LoRA r=8 **0.9143**
→ progressive unfreezing of the last 4 blocks **0.9189** → full fine-tune **0.8880**. Full
fine-tuning is *worse than training no backbone weights at all*, with 1,031× more trainable
parameters — 2,636 training images cannot support 304M. Raising input resolution to 448 px
also hurts (−0.009 QWK), because RETFound was pretrained at 224.

See `results/aptos_final_table.md` for the full ablation (including the asymmetric-loss, KC-loss
and non-uniform-squash negatives) and `results/literature_comparison_table.md` for a
side-by-side against published DR methods.

## Install

Python 3.12, managed with [`uv`](https://github.com/astral-sh/uv).

```bash
git clone https://github.com/vineetkumar-sudo/retfound-capsnet.git
cd retfound-capsnet
uv sync
```

This installs torch, torchvision, timm, peft, scikit-learn, matplotlib, wandb, and the Kaggle CLI into `.venv/`. All commands below are invoked via `uv run` so they pick up the project environment automatically.

## Datasets and weights

All raw datasets are gitignored; you need to obtain them yourself.

### APTOS-2019

Kaggle competition (research use only, no redistribution):

```bash
uv run kaggle competitions download -c aptos2019-blindness-detection -p data/aptos/
unzip -q data/aptos/aptos2019-blindness-detection.zip -d data/aptos/
```

Kaggle CLI authentication uses the legacy `~/.kaggle/kaggle.json` (not `KGAT_*` access tokens).

### IDRiD

Register at <https://idrid.grand-challenge.org/> and place the files so that the repo sees `data/idrid/B. Disease Grading/` with the official 413-image train set and 103-image test set.

### Messidor-2

1. Register at <https://www.adcis.net/en/third-party/messidor2/> for the raw images (ADCIS research-use licence, no redistribution). Place images under `data/messidor2/IMAGES/`.
2. Download the adjudicated 5-class DR grades from the Google Brain Kaggle release (<https://www.kaggle.com/google-brain/messidor2-dr-grades>) and place the label CSV under `data/messidor2/`.

### RETFound pretrained weights

```bash
mkdir -p data/weights
# Download RETFound_mae_natureCFP.pth from https://github.com/rmaphoh/RETFound_MAE
# and place it at data/weights/RETFound_mae_natureCFP.pth
```

The ViT-L nature-CFP MAE checkpoint is what the feature cache expects.

## Cache the frozen backbone features

Runs once per dataset; produces `data/<dataset>/features/*.npy` so that every downstream head trains on fixed feature matrices.

```bash
uv run python -c "from src.data.feature_cache import extract_aptos_features; extract_aptos_features()"
uv run python scripts/extract_idrid_features.py
uv run python scripts/extract_messidor2_features.py
```

## Reproduction

**One command reproduces the whole paper:**

```bash
bash scripts/regenerate_all.sh
```

That builds the 256/512 px image caches, extracts features at 224 and 448 for both
backbones, trains all six backbone × tune-mode tiers plus the Messidor-2 and IDRiD arms,
and then runs all ten metric scripts. It expects one A100-class GPU; the frozen tiers alone
run fine on CPU or MPS.

The rest of this section is the same pipeline broken into individually runnable steps. Each
"Day N" experiment is a self-contained script in `scripts/`. Every command writes its numeric
artifacts into `results/<experiment>/` and, for multi-seed runs, into
`results/<experiment>/seed{42,123,456}/`.

### APTOS (Days 1–5, 11)

```bash
# Day 1 — linear / MLP baselines (CE, MSE, weighted CE)
uv run python scripts/run_baselines.py --seeds 42,123,456

# Day 2 — vanilla 5-class CapsNet + margin loss
for s in 42 123 456; do uv run python scripts/run_capsnet.py --model-seed $s --output-dir results/capsnet/seed$s; done

# Day 3 — K-1 ordinal CapsNet (frozen, champion)
uv run python scripts/run_ordinal_capsnet.py --seeds 42,123,456

# Day 4 — asymmetric ordinal loss
uv run python scripts/run_asymmetric_ordinal.py --seeds 42,123,456

# Day 5A — KC Loss (differentiable QWK surrogate)
uv run python scripts/run_ordinal_kc.py --seeds 42,123,456

# Day 11 — non-uniform squash ablation
uv run python scripts/run_ordinal_capsnet.py \
    --config configs/ordinal_capsnet_nonuniform.yaml \
    --seeds 42,123,456 --no-wandb
```

### APTOS LoRA fine-tune

```bash
uv run python scripts/run_lora_ordinal.py --seeds 42,123,456 --resume
```

`--resume` is a footgun on a fresh clone: it reads the tracked `summary.json` *before* it globs for `preds_fold{N}.npz`, so it skips every fold and exits in seconds while printing the numbers from that summary. Pass `--force-redo`, or delete the stale `summary.json` files first. Total wall-clock is ~26 min per 5-fold run on an A100, or ~16 h on Apple M-series MPS.

### IDRiD (Day 8)

```bash
uv run python scripts/run_idrid.py
uv run python scripts/cross_dataset_aptos_to_idrid.py
```

### Messidor-2 (Day 9)

```bash
uv run python scripts/run_messidor2.py
uv run python scripts/cross_dataset_aptos_to_messidor2.py
uv run python scripts/posthoc_rank_calibration_messidor2.py
```

### Aggregated tables and paper figures

```bash
# Aggregate per-day tables into the paper-ready ablations
uv run python scripts/aggregate_day6.py
uv run python scripts/aggregate_day8.py
uv run python scripts/aggregate_day9.py

# Regenerate every figure referenced in the paper
uv run python scripts/make_day6_figures.py --strict
uv run python scripts/make_day8_figures.py
uv run python scripts/make_day9_figures.py --strict
uv run python scripts/make_lora_figures.py --strict
uv run python scripts/make_lora_rampup_figure.py
uv run python scripts/make_grade_samples_figure.py
uv run python scripts/make_architecture_diagram.py

# Compile the paper (requires pdflatex in TeX Live 2020+)
cd paper/submission && pdflatex article.tex && pdflatex article.tex
```

### Metrics and audit scripts

These need no training — they read the cached per-fold predictions and take seconds on CPU.

```bash
uv run python scripts/compute_calibration_metrics.py    # ECE, MCE, AURC, risk-coverage
uv run python scripts/compute_extended_calibration.py   # Brier, NLL, temperature scaling
uv run python scripts/compute_monotone_decode.py        # antitonic projection vs product decode
uv run python scripts/compute_loss_calibration.py       # loss x head factorial
uv run python scripts/compute_conformal_metrics.py      # LAC / APS split conformal
uv run python scripts/compute_ordinal_conformal.py      # RAPS, Mondrian CCP, OCP, set contiguity
uv run python scripts/compute_clinical_metrics.py       # referable / vision-threatening DR
uv run python scripts/compute_uda_baselines.py          # CORAL, BBSE, EM vs rank calibration
uv run python scripts/compute_significance_tests.py     # paired Wilcoxon across folds
uv run python scripts/profile_compute.py                # FLOPs, peak memory, throughput (needs CUDA)
```

## Repository structure

```
retfound-capsnet/
├── src/
│   ├── models/
│   │   ├── baselines.py          # Linear + MLP heads
│   │   ├── capsnet.py            # Vanilla 5-class CapsNet + routing history hook
│   │   └── ordinal_capsnet.py    # Shared PrimaryCaps + K-1 binary DigitCaps heads
│   ├── losses/
│   │   ├── margin_loss.py        # Sabour margin loss
│   │   ├── ordinal_loss.py       # K-1 decomposition + predict_grade_from_heads
│   │   ├── asymmetric_loss.py    # Direction-aware variant
│   │   └── kc_loss.py            # Differentiable QWK surrogate
│   ├── data/
│   │   └── feature_cache.py      # RETFound forward pass → .npy cache
│   ├── evaluate.py               # compute_all_metrics (QWK, Acc, F1, MAE, CM)
│   └── uncertainty.py            # Prediction margin, DigitCap entropy, routing variance
├── scripts/                      # One run_*.py / make_*.py / aggregate_*.py per day
├── configs/                      # YAML hyperparameters
├── results/                      # Lightweight artifacts (tables, CSVs, JSONs, figures)
├── paper/
│   └── submission/
│       ├── article.tex           # IEEEtran draft
│       ├── article.pdf           # Compiled paper
│       └── figs/                 # PDFs of every figure referenced in the draft
├── data/                         # Gitignored; raw datasets + RETFound weights live here
├── experiments/                  # Gitignored; wandb run logs, legacy checkpoints
├── LICENSE                       # MIT for code; third-party assets retain upstream licences
├── CLAUDE.md                     # Project-local notes (evaluation protocol, findings)
├── pyproject.toml                # uv / Python 3.12 environment
└── README.md                     # this file
```

Checkpoints (`*.pt`, `*.pth`, `*.ckpt`) and cached backbone features (`data/**/*.npy`) are
gitignored. Everything in `results/` is tracked: markdown tables, CSVs, JSON summaries,
PNG/PDF figures, **and the 413 per-fold prediction arrays** (`preds_fold*.npz`, 10 MB total).
Those arrays are the evidence base for every number in the paper — each holds `y_true`,
`y_pred` and the raw `head_probs`/`head_lengths` for one fold — so you can re-derive or
audit any table cell without re-running 20+ hours of training. That is also how the metric
scripts above run in seconds.

## Uncertainty signals — the honest audit

This is where the submitted version of the paper was wrong, and the corrected findings are
the main reason this repository exists.

- **The capsule margin is a valid ranking score, not a calibrated probability.** As a
  selective-prediction signal it is statistically significant (Mann–Whitney U, p < 10⁻³⁰⁰
  on the 3-seed APTOS pool) but it is *matched* by the top-2 margin of a plain MLP+K−1
  sigmoid head — AURC agrees within 0.002–0.004 in every cell. The "capsule-native UQ is a
  free strong signal" claim does not survive.
- **The capsule head is consistently worse calibrated than MLP+K−1** at comparable accuracy
  (APTOS ECE 0.143 vs 0.098). LoRA lifts accuracy to 0.824 but does not fix calibration.
- **The heads are under-confident, not over-confident.** Accuracy exceeds mean confidence in
  every equal-mass bin of all nine configurations, and the fitted temperature is below 1.
- **Most of the miscalibration is the loss, not the architecture — but not all of it.** A
  loss × head factorial shows that replacing the Sabour margin loss with plain BCE improves
  ECE by −0.079 (capsule) and −0.087 (MLP); at matched loss the capsule head is *still*
  worse by +0.032 to +0.061. Best cell is MLP+K−1 with BCE at ECE **0.0181** against 0.1581
  for capsule+margin — 8.7× better calibrated for −0.0065 QWK.
- **Ordinal coherence is a free fix.** The raw heads violate rank monotonicity on 4.2%–62.2%
  of samples. Projecting onto the antitonic cone ∩ unit box (exact, via PAVA, O(K)) then
  decoding by telescoping differences roughly **halves ECE** everywhere, with no held-out
  split and no fitted parameter. See `src/calibration.py`.
- **Conformal prediction holds up; class-conditional coverage does not.** Marginal coverage
  is within ±0.006 of the 90% target everywhere regardless of head miscalibration, but
  worst-class coverage falls to 0.583 under APS. LAC/APS/RAPS also emit *non-contiguous*
  grade sets 0.6%–12.8% of the time, which is meaningless on an ordinal scale; our **OCP**
  score removes all of them for +0.04 mean set size. See `src/conformal.py`.
- **Routing-agreement variance is an inverted signal** — rejecting high-"uncertainty"
  samples *lowers* accuracy on APTOS. Reported as a negative result, not hidden.

## Cross-dataset and post-hoc calibration

**APTOS → Messidor-2** collapses to QWK 0.030 because the RETFound CLS feature distribution
shifts: mean P(y>0) drops 0.51 → 0.087, so every head fires "No" and 99.4% of images are
predicted Grade 0. Rank order survives (binary AUC 0.620), so this is a *threshold* failure,
not a feature-space one. A one-line rank calibration against the APTOS training prior — no
target labels — recovers QWK to **0.240** and referable sensitivity from 2.0% to **53.8%**
(oracle prior: 0.278). It stays in the Landis–Koch *fair* band, well below clinical
agreement, so we report it as a diagnostic probe rather than a deployment recipe. We also
tested what reviewers suggested instead: CORAL feature alignment reaches only 0.174, and
posterior reweighting (BBSE, Saerens EM) fails outright at 0.049 and 0.0005 because the
decoded posterior has saturated at P(y=0) ≈ 1 and multiplicative reweighting cannot move the
arg-max. See `scripts/posthoc_rank_calibration_messidor2.py`, `scripts/compute_uda_baselines.py`
and `results/uda_table.md`.

**APTOS → IDRiD** tells the opposite story, and it is the reason the pessimistic reading of
cross-dataset transfer is specific to RETFound-MAE rather than general: the same transfer
rises from QWK 0.279 under RETFound to **0.626** under DINOv2 — within noise of the 0.657
obtained by training on IDRiD directly.

## Hardware

- **Canonical hardware is a single NVIDIA A100-SXM4-40GB.** Every number in the paper comes
  from that machine, driven end-to-end by `bash scripts/regenerate_all.sh`. Wall-clock for
  the full 5-fold APTOS run: 2 min frozen, 26 min LoRA r=8, 26 min progressive, 54 min full
  fine-tune.
- **The frozen tier genuinely runs on a laptop**: full 5-fold APTOS CV in ~2 min on an Apple
  M-series MacBook (MPS). This is the compute-efficient tier and it needs no GPU.
- **The fine-tuned tiers do not.** LoRA takes ~16 h for 3 seeds × 5 folds on MPS against
  ~26 min on the A100.
- Everything is implemented for CPU, CUDA, and MPS. Results across backends agree to within
  seed noise for the frozen tier, but the published numbers are the CUDA ones.

## Citation

If you use this code, please cite the paper draft:

```bibtex
@article{kumar2026trustworthydr,
  title   = {Trustworthy AI for Diabetic Retinopathy Grading: Auditing Foundation
             Models, Capsule Networks, and Conformal Uncertainty},
  author  = {Kumar, Vineet},
  year    = {2026},
  note    = {Draft. Code: https://github.com/vineetkumar-sudo/retfound-capsnet}
}
```

Please also cite the upstream works the method builds on:

- RETFound (Zhou et al., *Nature* 2023) for the pretrained ViT-L backbone.
- Sabour, Frosst, and Hinton (NeurIPS 2017) for the capsule mechanism and margin loss.
- Niu et al. (CVPR 2016) for the K−1 binary ordinal decomposition.
- Hu et al. (ICLR 2022) for LoRA.
- APTOS-2019 (Kaggle 2019), IDRiD (Porwal et al. 2018), and Messidor-2 (Decencière et al. 2014; Abràmoff et al. 2013; Krause et al. 2018) for the datasets.

Full BibTeX for the Messidor-2 citations is in `results/messidor2_acknowledgement.md`.

## Licence

Source code in this repository is released under the MIT License (`LICENSE`). Third-party assets—APTOS/IDRiD/Messidor-2 images, Messidor-2 adjudicated labels, and RETFound pretrained weights—retain their upstream research-use licences, which take precedence for those specific assets. See `LICENSE` for the full attribution text.
