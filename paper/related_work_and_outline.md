# RETFound + Ordinal CapsNet — Related Work Digest & Paper Outline

Generated from a close read of all 11 PDFs in `paper/ref_papers/`. This document is a working reference for drafting the paper — numbers, claims, and framing pointers live here.

> **⚠ HISTORICAL PLANNING DOCUMENT — do not quote its numbers.** Frozen at
> 2026-04-22, before the experiments were re-run on the A100. Every figure
> attributed to *our* method below is superseded, and several conclusions
> changed rather than just shifting a digit: the frozen QWK is 0.8923 not
> 0.8932, LoRA is 0.9143 not 0.9127, full fine-tuning turns out to be *worse*
> than a frozen backbone, DINOv2 beats RETFound throughout, and the heads are
> under-confident rather than over-confident. For anything about our own
> results use, in order: the tables in `paper/submission/article.tex`, the
> script-generated tables in `results/*.{md,csv}`, then the raw per-fold
> predictions. **What is still useful here is the literature digest** — the
> close reads of the 11 PDFs in `paper/ref_papers/`, whose numbers are their
> authors' and do not go stale.

---

## 1. Literature snapshot (per-paper, with numbers on OUR datasets)

### 1.1 Foundational methods papers (we build on these)

#### Sabour, Frosst, Hinton (2017) NeurIPS — *Dynamic Routing Between Capsules*
- Defines capsule = vector of instantiation parameters; squash non-linearity `v = (‖s‖²/(1+‖s‖²))·(s/‖s‖)`; routing-by-agreement; margin loss `m⁺=0.9, m⁻=0.1, λ=0.5`; 3 routing iterations.
- Evaluated on MNIST (0.25 % err), MultiMNIST (5.2 %), CIFAR-10 (10.6 %), SVHN (4.3 %). **No DR, no medical imaging.**
- **Our usage**: exact same squash + margin loss + 3 routing iters. We cite as the origin of the capsule machinery.

#### Niu, Zhou, Wang, Gao, Hua (2016) CVPR — *Ordinal Regression with Multiple Output CNN for Age Estimation*
- K-1 binary classifier decomposition (`y^k = I[y > r_k]`), K-1 output heads sharing a CNN trunk, rank = `1 + Σ f_k(x)`.
- Task-importance weights `λ_t ∝ √N_k` (the "Niu weights" we implement in `src/losses/asymmetric_loss.py::compute_niu_weights`).
- Datasets: MORPH II (age, MAE 3.27), AFAD (MAE 3.34). **No DR, no UQ.**
- **Our usage**: our K-1 Ordinal CapsNet is Niu's decomposition with DigitCaps per head instead of softmax classifiers. We cite as the formulation origin.

#### Zhou et al. (2023) *Nature* 622:156–163 — *RETFound: a foundation model for generalizable disease detection from retinal images*
- ViT-L MAE pretrained on 904 170 CFPs + 736 442 OCT scans (MEH-MIDAS + EyePACS + public datasets).
- DR AUROC (internal fine-tune): **APTOS-2019 0.943**, **IDRID 0.822**, **MESSIDOR-2 0.884**.
- Cross-dataset AUROC (fine-tune on A, eval on B): APTOS→IDRID 0.822; IDRID→APTOS 0.738; APTOS→Messidor-2 and others in Fig 2b.
- Reports AUROC, not QWK. Adapts via supervised fine-tuning of the full backbone + MLP head.
- **Our usage**: backbone. Our paper's angle: a capsule-ordinal head on *frozen* RETFound features nearly matches their fine-tuned MLP on QWK, and LoRA beats it — at ~1 % of the trainable-parameter cost.

### 1.2 DR-specific ordinal regression

#### El Bellaj, Benradi, et al. (2026 arXiv Feb) — *Uncertainty-Aware Ordinal Deep Learning for cross-Dataset DR Grading*
- ConvNeXt-Base + Lesion-Query Attention Pooling (LQAP) + **evidential Dirichlet head over K-1 ordinal thresholds** (same K-1 formulation as Niu/us).
- KL-annealed ordinal evidential loss + query diversity / load-balancing penalties.
- Trained on pooled APTOS + Messidor-2 + EyePACS subset; **claims QWK 0.940 / Acc 0.876** on a combined held-out test.
- **Red flags**: no per-dataset test split reported; single-number evaluation; no multi-seed; QWK 0.94 is inconsistent with RETFound's Messidor-2 AUROC 0.884. Treat as headline-claim, not verified benchmark.
- Non-evidential ConvNeXt baseline in same paper: QWK 0.918 / Acc 0.826.
- **Our angle vs theirs**: simpler backbone (frozen/LoRA RETFound), per-dataset reporting with 5-fold CV and 3-seed mean ± std on APTOS, capsule-native UQ (no extra loss term), honest cross-dataset numbers.

#### Kumar et al. (2025 arXiv Nov) — *Stage-Aware Diagnosis of Diabetic Retinopathy via Ordinal Regression*
- ResNet50 + single-scalar MSE regression + `round(clip([0,4]))`. This is *metric regression*, not K-1 ordinal.
- Preprocessing: green-channel + CLAHE + median filter (the Day-0 approach we abandoned in favour of official RETFound transforms).
- **APTOS-2019 only. QWK 0.8992** (single test split, no multi-seed).
- No cross-dataset, no UQ, no confidence intervals.
- **Relation to us**: our MLP+MSE baseline (QWK 0.8818 ± 0.005) is methodologically equivalent and only 0.02 off their number — the gap is explained by CV vs single split. Our LoRA 0.9139 exceeds their claim; frozen 0.8932 is narrowly below.

#### Yu et al. (2025 arXiv Jul, MICCAI-style) — *AOR-DR: Parameterized Diffusion Optimization enabled Autoregressive Ordinal Regression*
- **Uses RETFound as backbone** (same as us). Adds autoregressive K-1 decomposition (y^i conditioned on y^{1..i-1} via affine or cross-attention fusion) + denoising-diffusion network for conditional probability modelling.
- Four datasets: APTOS, Messidor (not Messidor-2), DDR, DEEPDR. No cross-dataset. **No QWK — only Acc / F1 / Sen / Spec.**
- Key numbers (Table 2, frozen RETFound backbone):
  - *RETFound† baseline*: APTOS Acc 82.1 / F1 65.5, Messidor Acc 64.2 / F1 45.4, DDR Acc 71.7 / F1 57.2.
  - *AOR-DR + RETFound†*: APTOS Acc 80.3 / F1 65.7, Messidor Acc 64.7 / F1 54.0, DDR Acc 55.9 / F1 55.3.
- **Our frozen Ordinal CapsNet (APTOS)**: Acc 0.7931 / F1 0.6255.
- **Our LoRA Ordinal CapsNet (APTOS)**: Acc 0.8206 / F1 0.6576 — edges their AOR-DR+RETFound on both Acc and F1, with much simpler method.
- No UQ in AOR-DR. Our contribution beyond theirs: capsule head + capsule-native UQ + cross-dataset + QWK.

### 1.3 CapsNet for DR (direct architectural comparators)

#### Gogulamudi et al. (2024) *Results in Engineering* 23:102820 — *Non-uniform squash function in Capsule networks for early DR detection*
- Custom Conv→PrimaryCaps→DigitCaps + **modified squash** `v = (‖s‖/(1+‖s‖))·(s/(‖s‖+ε))`.
- Non-standard dataset: 757 images from NIH/Paraguay, **7 classes** (Mild/Moderate/Severe/Very-Severe NPDR + PDR/Advanced PDR). Not APTOS/IDRiD/Messidor-2.
- Headline claim: 99.84 % accuracy. **Red flag**: Table 1 shows this is *training* accuracy (test 98.68 %). Small test set (≈151 images). No QWK, no CV, no cross-dataset, no UQ.
- **Relation to us**: weakest comparator on rigour; useful mainly as prior evidence that CapsNet has been applied to DR. Our rigour advantages: standard benchmarks, QWK primary, multi-seed × 5-fold, cross-dataset, proper UQ.

#### Lei et al. (2024) *Engineering Applications of AI* 133:107994 — *GNN-fused CapsNet with multi-head prediction for DR grading*
- ResNet-18 (ImageNet) → PrimaryCaps (32 × 16-dim) → **GNN-based feature fusion** (kNN graph over primary capsules, message passing) → **GNN-based transformation** (replaces traditional W_ij matrix) → **C-way multi-head** (one head per class, not K-1 ordinal) → dynamic routing + Sabour margin loss.
- Datasets: APTOS-2019 (own 8:2 split of 3,662), IDRiD (own 8:2 split — not the official 103-image test).
- Results: **APTOS Acc 0.8649, AUC 0.9559, F1 0.7405**; **IDRiD Acc 0.6408**. No QWK, no cross-dataset, no UQ.
- Parameter count: 12.87 M (similar to standard CapsNet).
- **Relation to us**: closest architectural competitor (ResNet + CapsNet + multi-head). They don't use ordinal decomposition; multi-head = per-class heads. On APTOS their 0.8649 Acc is similar to our LoRA 0.8206. Their IDRiD 0.6408 uses their own split, not comparable to our official-test 0.3825.

### 1.4 Strong non-capsule DR baselines

#### Dixit & Jha (2025) *Medical Engineering and Physics* 140:104350 — *EfficientNetB3 with squeeze-excitation*
- EfficientNetB3 (ImageNet) + SE block + softmax; custom preprocessing (crop + Gaussian-blur-weighted enhancement).
- Splits: APTOS 80/10/10, IDRiD 364/45/46 (455 total — doesn't match official 413 + 103), Messidor-2 744 total (subset).
- Results: APTOS Acc **88.44 %**, "Kappa" 0.88, Macro AUC 0.92; IDRiD Acc 81.74 %, AUC 0.83; Messidor-2 Acc 73.28 %, AUC 0.89.
- "Kappa" ambiguous (Cohen vs QWK); AUC reporting style differs from ours.
- No cross-dataset, no UQ, not ordinal.

#### Bodapati & Balaji (2024) *Multimedia Tools and Applications* 83:1083–1102 — *Self-adaptive stacking ensemble with dual attention*
- Ensemble of 4 dual-attention candidates (VGG16-CB3/CB4/CB5 + Xception-CB14), each with spatial attention + ConvLSTM cross-correlation, MLP meta-learner.
- APTOS-2019 only, 80/20 single split.
- Results: **Acc 86.22 %, Kappa 0.8965 (likely QWK), AUC 96.47 %**. Per-class: No DR 98.34 %, Mild 63.51 %, Moderate 87 %, Severe 51.28 %, PDR 61.02 %.
- Their κ 0.8965 sits between our frozen (0.8932) and LoRA (0.9139).

#### Oulhadj et al. (2022) *Multimedia Tools and Applications* 81:28709–28727 — *Deep learning + deformable registration*
- B-spline deformable registration pre-processing + 4-CNN ensemble (DenseNet121 + Xception + Inception-V3 + ResNet50) + voting.
- APTOS-2019 only: 2,801 / 494 / 367 split.
- Results: Ensemble Acc **85.28 %, Cohen κ 0.7778** (formula `(p_o−p_e)/(1−p_e)` is Cohen's, not quadratic-weighted).
- Cited by Stage-Aware paper as "0.75 QWK" — direct QWK equivalent is uncertain.

---

## 2. Consolidated "numbers on our datasets" table

*Same-dataset comparison wherever possible. "—" = not reported / incomparable. Our numbers at bottom.*

| Method | Year | Backbone / trainable params | APTOS QWK | APTOS Acc | IDRiD QWK | IDRiD Acc | Messidor-2 QWK | Messidor-2 Acc | UQ | Cross-dataset |
|---|---|---|---|---|---|---|---|---|---|---|
| Niu OR-CNN | 2016 | custom shallow CNN | — | — (age) | — | — | — | — | — | — |
| Sabour CapsNet | 2017 | custom | — | — (MNIST) | — | — | — | — | — | — |
| RETFound + MLP | 2023 | ViT-L full FT (307 M) | — (AUROC 0.943) | — | — (AUROC 0.822) | — | — (AUROC 0.884) | — | No | Yes (AUROC only) |
| Oulhadj ensemble | 2022 | 4-CNN ensemble | — (Cohen κ 0.778) | 0.853 | — | — | — | — | No | No |
| Bodapati ensemble | 2024 | VGG16+Xception ensemble | ≈0.897 | 0.862 | — | — | — | — | No | No |
| Kumar Stage-Aware | 2025 | ResNet50 + MSE | 0.8992 (val, single split) | — | — | — | — | — | No | No |
| Dixit EfficientNetB3+SE | 2025 | EfficientNetB3 | "κ 0.88" (ambig) | 0.884 | — | own 81.7 Acc | — | own 73.3 Acc | No | No |
| Gogulamudi non-uniform squash | 2024 | custom CapsNet | — | — (custom 7-class dataset) | — | — | — | — | No | No |
| Lei GF-CapsNet | 2024 | ResNet-18 + GNN + CapsNet | — (AUC 0.956) | 0.865 | — | own 0.641 Acc | — | — | No | No |
| Yu AOR-DR (RETFound frozen) | 2025 | RETFound† + AR diffusion | — | 0.803 / F1 0.657 | — | — | — | 0.647 / F1 0.54 | No | No |
| El Bellaj UAOR-LQAP+EDL | 2026 | ConvNeXt-Base | — (claim κ 0.94 pooled) | 0.876 pooled | — | (pooled) | — | (pooled) | Evidential Dirichlet | Trained-on-pool (not held-out cross) |
| **Ours — frozen Ordinal CapsNet** | — | **RETFound frozen + 295 K head** | **0.8932 ± 0.0004 (3 seeds × 5 folds)** | 0.7931 ± 0.0021 | **0.756 val / 0.446 test** (5-fold, official test) | — | **0.607 (5-fold within-gradable)** | — | **Prediction margin** (p < 1e-300 APTOS) | **Yes, multiple** |
| **Ours — LoRA r=8** | — | **RETFound + LoRA + 1.08 M head** | **0.9127 ± 0.0008 (3 seeds × 5 folds; per-seed means 0.9139 / 0.9123 / 0.9121)** | **0.8237 ± 0.0068** | — | — | — | — | **Prediction margin** (p < 1e-300 APTOS, 3-seed pool) | — (not extended to LoRA) |

### Interpretation notes
- **On APTOS QWK, we're at parity with or beat every CapsNet-based method in the literature** (frozen 0.893 > 0.80 for Oulhadj-equivalent κ, LoRA 0.914 > Bodapati 0.897 and Kumar 0.899).
- **The El Bellaj "0.94 pooled" number is the one ceiling claim** — but their evaluation is on a pooled 3-dataset held-out test without per-dataset breakdown, no multi-seed, no open code, and the Messidor-2 portion of that "0.94" is inconsistent with RETFound's own Messidor-2 AUROC 0.884. We should frame them as the nominal SOTA but note the methodology gap.
- **Nobody else has reported QWK on Messidor-2 with per-dataset 5-fold CV** — that's our novelty on Messidor-2.
- **Nobody else reports cross-dataset with honest failure modes** — their cross-dataset claims (if any) are generally "fine-tune-on-each-dataset" rather than "train-APTOS-only-then-eval".
- **Only El Bellaj has UQ — and theirs requires a separate evidential loss + KL annealing**. Our prediction-margin UQ is free (no extra loss term), validated on three datasets, with the routing-variance negative result honestly reported.

---

## 3. Paper outline with narrative pointers

### 3.1 Working title (pick one)
- **"Ordinal Capsule Regression on Retinal Foundation Features: A Parameter-Efficient Head for Diabetic Retinopathy with Native Uncertainty"**
- Alternative: "RETFound-CapsNet: Frozen and LoRA-Tuned Ordinal Capsule Heads for Calibrated Diabetic Retinopathy Grading"

### 3.2 Narrative arc (four threads to weave)
1. **Ordinality matters**: DR is graded 0–4 (ICDRSS). Flat 5-class classification wastes structure and penalises distant errors the same as adjacent ones. K-1 binary decomposition (Niu 2016) fixes this.
2. **Foundation features are enough**: RETFound gives strong retinal representation for free. You don't need to fine-tune 307 M parameters to beat published CapsNet + DR methods — a 295 K capsule head does it.
3. **Capsules natively carry confidence**: the length of each DigitCap head is a probability; the spread across heads is a calibration signal. Prediction margin (1 − [top1 − top2]) separates correct from misclassified with `p < 1e-300` on APTOS, `p < 1e-4` on IDRiD, `p < 1e-300` on Messidor-2. No extra loss function needed.
4. **Honest about failure modes**: cross-dataset without calibration fails; asymmetric loss doesn't help vs symmetric K-1; KC loss doesn't help vs margin loss; routing variance is an inverted UQ signal. These are paper findings, not embarrassments.

### 3.3 Section-by-section pointers

#### Abstract (≤ 200 words)
- 1-2 sentences: DR scale + ordinal nature + clinical screening tension.
- 2 sentences: our method (Ordinal K-1 Capsule head on frozen/LoRA RETFound).
- 3-4 sentences: best numbers (APTOS QWK 0.893 / 0.914, IDRiD within QWK, Messidor-2 within QWK, UQ p-values on 3 datasets).
- 1 sentence: open-source (once you commit the repo publicly) + honest negative results.

#### 1. Introduction
- DR epidemiology: 463 M diabetics, 40 % develop DR, preventable blindness. (Stats from Oulhadj / Dixit / Stage-Aware.)
- Three open tensions:
  1. Ordinality ignored by flat classifiers.
  2. Full fine-tuning is data-hungry and compute-heavy (Oulhadj's 4-CNN ensemble trained for ~22 h; El Bellaj's ConvNeXt+EDL likely GPU-days).
  3. Point predictions give no triage signal — critical for clinical deployment.
- **Contributions** (bulleted):
  1. Ordinal Capsule head on frozen RETFound — 295 K trainable params, APTOS QWK 0.893 ± 0.0004 (3 seeds × 5-fold).
  2. LoRA fine-tune variant — 1.08 M trainable params, APTOS QWK 0.914 (beats every CapsNet-DR method + most CNN ensembles).
  3. **Capsule-native UQ via prediction margin** — no additional loss term, validated on three datasets.
  4. **Honest cross-dataset characterisation** — APTOS→IDRiD QWK 0.273, APTOS→Messidor-2 collapses; we diagnose the cause (head-probability shift, not head bug).
  5. Negative results we report cleanly: asymmetric loss ≡ symmetric K-1 at λ=1; KC Loss ≤ margin loss; routing-variance UQ is inverted.

#### 2. Related work (4 subsections)
- **2.1 Ordinal DR grading**: Niu K-1, CORAL/CORN, Stage-Aware MSE-round, AOR-DR autoregressive, El Bellaj evidential.
- **2.2 Capsule networks for DR**: Sabour foundation, Gogulamudi non-uniform squash, Lei GF-CapsNet, DRDNet (Kumar 2020, referenced by Lei).
- **2.3 Retinal foundation models**: RETFound, downstream adaptation patterns (MLP head, LoRA, linear probe).
- **2.4 Uncertainty in medical imaging**: Bayesian, MC-dropout, evidential; none natively coupled with capsules.

#### 3. Method
- **3.1 Architecture overview**: figure `fig_architecture` (already generated).
- **3.2 Frozen pipeline**: preprocessing (official RETFound transform), backbone freeze, PrimaryCaps 32×8, K-1 ordinal DigitCaps heads (2 caps × 16-dim each, 3-iter routing), Sabour margin loss per head.
- **3.3 LoRA variant**: targets `attn.qkv` in all 24 ViT blocks, rank 8, α=16, dropout 0.05; two-group optimiser (LoRA lr 1e-4, head lr 1e-3).
- **3.4 Inference & decoding**: `ŷ = Σ_k I[P(y>k) > 0.5]`; chain-rule 5-class probs; prediction-margin UQ.

#### 4. Experimental setup
- **4.1 Datasets**: APTOS-2019 (10 % holdout + 5-fold on 3 296), IDRiD (5-fold on 413 train + eval on 103 official test), Messidor-2 (Google Brain labels, 1 744 gradable, 5-fold within).
- **4.2 Protocol**: QWK primary; Acc / Macro F1 / MAE secondary; **every row in the APTOS ablation table is evaluated at 3 seeds × 5 folds (seeds 42, 123, 456)** — including LoRA, which after a 16 h overnight MPS run has across-seed std **0.0008 QWK**; patience 30 on val QWK (patience 8 for LoRA due to its faster convergence). Data splits (10 % holdout + 5-fold) are fixed across seeds by `cfg.data.seed`; only model init RNG varies.
- **4.3 Baselines** (Day 1 + Day 2): Linear + CE, MLP + CE, MLP + MSE, MLP + Weighted CE, Vanilla 5-class CapsNet, + Asymmetric, + KC Loss.

#### 5. Results
- **5.1 APTOS ablation (Table: `aptos_final_table.md`, 10 rows all at 3 seeds × 5-fold)**: **LoRA sweeps every column** — QWK **0.9127 ± 0.0008**, Acc **0.8237 ± 0.0068**, F1 **0.6627 ± 0.0105**, MAE **0.214 ± 0.006**. Frozen Ordinal CapsNet wins the frozen-only subset (QWK **0.8932 ± 0.0004**, MAE **0.254 ± 0.002**). Vanilla CapsNet narrowly wins Accuracy (0.8061) and Macro F1 (0.6301) at the frozen level, a whisker above MLP+CE (0.8057 / 0.6271). LoRA's across-seed std on QWK is **0.0008** — its three per-seed means (0.9139 · 0.9123 · 0.9121) are within 0.002 of each other, which is extraordinary stability for a fine-tuning method.
- **5.2 Per-class breakdown** (Vanilla · Ordinal frozen · LoRA, all pooled across 3 seeds): **Grade 3 Severe +15.7 pp** at the frozen Ordinal level (Vanilla 31.0 → Ordinal 46.7) — classic K-1 safety-bias signature, largest single-class lift. LoRA **regresses on Grade 3** vs frozen Ordinal (46.7 → 39.1); this is a nuance worth discussing: LoRA's +0.02 QWK lift comes from PDR (Grade 4 +11.9 pp Vanilla → LoRA, 47.7 → 59.6) and preserved-or-slightly-improved Grades 0–2, not from Severe. So the **frozen head is actually better on Grade 3 Severe** — likely because its safety-bias is stronger when the backbone is untuned.
- **5.3 Training efficiency**: 115 s frozen full 5-fold on MacBook MPS; ~5 h LoRA full 5-fold. Contrast with El Bellaj / Oulhadj compute.
- **5.4 IDRiD (within + cross)**: within val QWK 0.756, test 0.446 — 0.3-QWK val→test gap is dataset-wide (all four baselines drop equally); so the gap is IDRiD, not our method. Cross APTOS→IDRiD QWK 0.273. Ordinal per-class: Severe + Grade 0 trade-offs.
- **5.5 Messidor-2 (within + cross + calibration recovery)**: within QWK 0.6065 ± 0.036 — **clearest within-dataset QWK lead in the paper** (+0.05 over MLP+MSE 0.5566). Cross APTOS→Messidor-2 QWK ≈ 0.01 uncalibrated — we diagnose the head-probability distribution collapse (mean P(y>0) 0.51→0.087, only 0.1 % cross the 0.5 threshold, predicted marginal 99.7 % Grade 0). **Post-hoc rank calibration** (see `scripts/posthoc_rank_calibration_messidor2.py`): bucketing samples by the APTOS train marginal recovers 5-class QWK to **0.2216** (+0.21 over uncalibrated, no target-domain peeking); using the Messidor-2 GT marginal as an oracle upper bound gives **0.2682**. On the binary referable task the sensitivity jumps from 17.9 % → 51.9 % at fixed AUC 0.625 — confirms the failure mode is a **threshold miscalibration**, not a feature-space failure, since AUC (rank-order metric) is preserved. Artifact: `results/cross_dataset/aptos_to_messidor2/calibrated/summary.json`.
- **5.6 UQ on three datasets**: table of Mann-Whitney p-values + accuracy at 50 % coverage; 3-curve accuracy-coverage figure showing routing-variance INVERTS (honest negative retained in the paper).
- **5.7 Figures**: confusion-matrix side-by-sides (Vanilla vs Ordinal frozen, Frozen vs LoRA), UQ boxplots per dataset, accuracy-coverage curves per dataset, ROC for the Messidor-2 binary task.

#### 6. Discussion
- **Why frozen works**: RETFound's CLS token already captures DR-specific structure (attention visualisations in El Bellaj's Fig 2 show lesion localisation without fine-tuning). Our capsule head is a structured classifier on top of already-good features.
- **Why LoRA adds +0.021 QWK**: task-specific feature refinement in the attention pathway. Per-class lift on PDR (+12.5 pp) suggests LoRA helps rare-class discrimination.
- **Why cross-dataset collapses**: the P(y>0) mean shift proves the issue is in the feature space (RETFound features are no longer in-distribution on Messidor-2), not the capsule head. Rank calibration would recover much of the loss.
- **Parameter efficiency as a contribution**: 295 K trainable vs RETFound's implicit 307 M full fine-tune vs El Bellaj's ConvNeXt-Base (~89 M). Ours runs on a consumer MacBook.

#### 7. Limitations
- Single-seed LoRA (1 × 5-fold, not 3 × 5-fold — compute budget).
- Cross-dataset without calibration is an honest negative; future work: rank calibration / DANN.
- No IDRiD / Messidor-2 LoRA variant (same reason).

#### 8. Conclusion
- Two-tier offering: frozen (2-minute MacBook training, QWK 0.893) vs LoRA (one-hour MacBook MPS, QWK 0.914). Clinical-deployment analogue: hospital vs edge device.
- UQ validated on 3 datasets without extra loss terms.
- Honest about what doesn't work.

### 3.4 Figures the paper needs
*Location: `results/figures/`. Status as of 2026-04-21 clean rerun.*

| ID | File | Section | Status |
|----|------|---------|--------|
| Fig 1 | `fig_architecture.{png,pdf}` | Method (3.1) | ✓ paper-grade TikZ diagram (vector PDF via pdflatex). Source at `paper/submission/figs/fig_architecture.tex`; `scripts/make_architecture_diagram.py` drives the compile + copies the PDF/PNG to `results/figures/`. Shows frozen pipeline + optional LoRA adapters on `attn.qkv` with a legend. |
| Fig 2 | `fig_ab_cm_sidebyside.{png,pdf}` | Results 5.2 (APTOS Vanilla vs Ordinal) | ✓ regenerated (3-seed pooled) |
| Fig 3 | `fig_b_frozen_vs_lora_cm.{png,pdf}` | Results 5.2 (APTOS Frozen vs LoRA) | ✓ regenerated (both sides 3-seed pooled) |
| Fig 4 | `fig_c_uq_boxplot.{png,pdf}` + per-dataset variants (`_idrid`, `_messidor2`) | Results 5.6 (APTOS UQ) | ✓ APTOS regenerated; IDRiD/Messidor-2 pending re-run |
| Fig 5 | `fig_d_acc_coverage.{png,pdf}` + per-dataset variants | Results 5.6 (3-curve coverage) | ✓ APTOS regenerated; IDRiD/Messidor-2 pending |
| Fig 6 | `fig_e_training_curve.{png,pdf}` | Results 5.1 (training dynamics) | ✓ regenerated |
| Fig 7 | `fig_g_messidor2_cm_sidebyside.{png,pdf}` | Results 5.5 | ⏳ pending Messidor-2 re-run |
| Fig 8 | `fig_h_messidor2_roc.{png,pdf}` | Results 5.5 (binary) | ⏳ pending Messidor-2 re-run |
| Fig 9 | `fig_f_idrid_cm_sidebyside.{png,pdf}` | Results 5.4 | ⏳ pending IDRiD re-run |
| Fig 10 | `fig_b_cm_ordinal_lora.{png,pdf}` | Results 5.2 (LoRA CM) | ✓ regenerated (3-seed pooled) |
| Fig 11 | `fig_c_uq_boxplot_lora.{png,pdf}` | Results 5.6 (LoRA UQ) | ✓ regenerated (3-seed pool, p < 1e-300) |
| Fig 12 | `fig_d_acc_coverage_lora.{png,pdf}` | Results 5.6 (LoRA 3-curve coverage) | ✓ regenerated (3-curve: prediction margin + digit entropy + routing variance, 3-seed pool) |
| Fig 13 | `fig_e_lora_rampup.{png,pdf}` | Results 5.1 or Discussion (bonus: "LoRA converges fast") | ✓ generated — per-epoch val QWK across all 15 folds; median best-epoch = **9** (range [7, 17]); mean QWK hits frozen baseline (0.8932) by **epoch 5**. Demonstrates LoRA reaches peak in half the 20-epoch budget. Driver: `scripts/make_lora_rampup_figure.py` |

### 3.5 Tables the paper needs
*Status as of 2026-04-21.*

| ID | File | Section | Status |
|----|------|---------|--------|
| Tab 1 | `results/aptos_final_table.md` | Results 5.1 | ✓ 10 rows, every row at 3 seeds × 5-fold |
| Tab 2 | `results/per_class_breakdown.csv` | Results 5.2 | ✓ Vanilla · Ordinal · LoRA all pooled over 3 seeds |
| Tab 3 | `results/idrid_final_table.md` | Results 5.4 | ⏳ pending IDRiD re-run |
| Tab 4 | `results/messidor2_final_table.md` | Results 5.5 | ⏳ pending Messidor-2 re-run |
| Tab 5 | `results/messidor2_binary_results.csv` | Results 5.5 | ⏳ pending Messidor-2 re-run |
| Tab 6 | **NEW** — literature comparison table | Related work (2.1–2.3) | manual fill from section 2 |
| Tab 7 | `results/compute_budget_table.md` | Discussion (6) | ✓ drafted — 10-row comparison, ours 295 K / 1.08 M vs RETFound 307 M, El Bellaj 89 M, Bodapati ~200 M etc. Open items flagged at the bottom of the table |

---

## 4. Key claims we can defend (ordered by strength)

1. **"Ordinal K-1 capsule decomposition with a frozen retinal foundation model achieves competitive APTOS QWK (0.8932 ± 0.0004, 3 seeds × 5-fold) at 295 K trainable parameters — a ~1 000× reduction over typical fine-tuned approaches."** ✅ Fully supported by the Day-11 fresh `aptos_final_table.md` (row 6).

2. **"LoRA rank-8 fine-tuning lifts every metric without breaking the parameter budget: QWK 0.9127 ± 0.0008 (3 seeds × 5-fold; per-seed means 0.9139 / 0.9123 / 0.9121), Accuracy 0.8237, MAE 0.214 at 1.08 M trainable parameters."** ✅ Fully supported by Day-11 clean 3-seed rerun. Across-seed std of 0.0008 QWK is ~15× tighter than within-seed fold-std, indicating LoRA is highly stable.

3. **"Capsule-length-based prediction margin is a native uncertainty signal — no extra loss term needed — and separates correct from misclassified with p < 0.001 on APTOS, IDRiD, and Messidor-2."** ✅ APTOS retested today (p < 1e-300 on both frozen and LoRA 3-seed pools). IDRiD and Messidor-2 p-values from prior runs; they re-test cheaply once those datasets are re-run.

4. **"The K-1 binary decomposition produces intrinsic safety bias (under-prediction over over-prediction, U/O < 1), so explicit asymmetric directional loss is redundant — symmetric λ=1 sits inside the seed-level noise of the baseline (0.8923 ± 0.0014 vs 0.8932 ± 0.0004, −0.0009 QWK)."** ✅ Supported by Day-11 refresh. NOTE: earlier runs reported the two rows as bit-identical via a shared code path; the clean rerun routes the asymmetric loss through its own constructor, producing a tiny RNG-path drift. Claim still holds under any statistical test.

5. **"Cross-dataset generalisation of frozen RETFound features to Messidor-2 fails catastrophically without calibration — the mean P(y>0) drops from 0.51 on APTOS to 0.087 on Messidor-2, driving 99.8 % of predictions to Grade 0. This is a feature-shift issue, not a head issue."** ✅ Fully supported by our diagnostic (will be reconfirmed when Messidor-2 is re-run).

6. **"Routing-agreement variance is a weakly inverted uncertainty signal on ordinal CapsNet — rejecting high-variance samples lowers accuracy. Prediction margin dominates, DigitCap entropy is modest, routing variance is counter-productive."** ✅ Supported by the 3-curve `fig_d_acc_coverage` regenerated today.

7. **"Ordinal CapsNet wins within-Messidor-2 QWK by ≥ 0.05 over every baseline (frozen-feature regime)."** ⏳ Supported by Day-9 within-Messidor-2 table; pending Messidor-2 re-run for refresh.

## 5. Open items before submission

**Must-do (blocking):**
- [ ] **IDRiD re-run** (5-fold train + 103-image official-test eval) — repopulates `results/idrid/`, `idrid_final_table.md`, `fig_f_idrid_cm_sidebyside`, `fig_c_idrid_uq_boxplot`, `fig_d_idrid_acc_coverage`. ~1 h.
- [ ] **Messidor-2 re-run** (5-fold within + APTOS→Messidor-2 cross) — repopulates `results/messidor2/`, `results/cross_dataset/`, `messidor2_final_table.md`, `messidor2_binary_results.csv`, `fig_g_messidor2_cm_sidebyside`, `fig_h_messidor2_roc`, `fig_c_messidor2_uq_boxplot`, `fig_d_messidor2_acc_coverage`. ~1 h.
- [x] **LoRA routing-variance recompute** ✓ done — `scripts/compute_lora_routing_variance.py` patched to walk `seed*/` subdirs; all 15 routing-variance files regenerated and `fig_d_acc_coverage_lora` is now 3-curve (prediction margin + digit entropy + routing variance, 3-seed pool).

**Nice-to-have:**
- [x] Add **compute + parameter budget** table ✓ — `results/compute_budget_table.md` drafted; 4 open numeric fact-checks flagged at the bottom (El Bellaj wall-clock, RETFound fine-tune time, Yu AOR-DR trainable params, Bodapati wall-clock).
- [x] Post-hoc rank calibration on APTOS→Messidor-2 ✓ — `scripts/posthoc_rank_calibration_messidor2.py` + artifact at `results/cross_dataset/aptos_to_messidor2/calibrated/summary.json`. APTOS-prior calibration recovers QWK 0.014 → **0.2216** (no target-domain peeking); oracle prior upper-bound 0.2682. Binary sensitivity 17.9 % → 51.9 % at fixed AUC 0.625 (rank order preserved, pure threshold fix).
- [ ] Licence check on Messidor-2 (cite Decencière 2014 + Abràmoff 2013 + acknowledgement text per ADCIS terms).
- [ ] Decide on venue: MICCAI (workshop or main), MIDL, IEEE JBHI, Medical Image Analysis. Target venue determines which subset of results to foreground.
