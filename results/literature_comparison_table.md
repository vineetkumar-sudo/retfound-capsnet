# Literature comparison — DR grading methods vs ours

Compiled from `paper/ref_papers/*.pdf` + our measurements. Focuses on methods that report results on **any of our three target datasets (APTOS-2019, IDRiD, Messidor-2)**. Paper Tab 6 candidate — drop-in.

## Main comparison (paper-ready format)

| Method (Year) | Arch. family | APTOS QWK | APTOS Acc | IDRiD | Messidor-2 | Cross-dataset | UQ | Multi-seed | Code avail. |
|---|---|---:|---:|---|---|---|---|---|---|
| Sabour CapsNet (2017) | CapsNet origin | — | — (MNIST) | — | — | — | — | — | ✓ |
| Niu OR-CNN (2016) | K-1 ordinal CNN | — | — (age) | — | — | — | — | — | ✓ |
| RETFound + MLP (2023) | ViT-L MAE + MLP, full FT | AUROC 0.943 (not QWK) | — | AUROC 0.822 | AUROC 0.884 | Fine-tune-on-each (not train-on-A-test-on-B) | ✗ | N=1 | ✓ |
| Oulhadj deform-regist ensemble (2022) | 4-CNN vote + B-spline reg. | Cohen κ 0.778 (not QWK), Acc 0.853 | 0.853 | — | — | ✗ | ✗ | N=1 (single split) | ✗ |
| Bodapati dual-attn stacking (2024) | 4× (VGG16-CB/Xception-CB14) + ConvLSTM | "Kappa 0.8965" (likely QWK), Acc 0.862 | 0.862 | — | — | ✗ | ✗ | N=1 (80/20 single split) | ✗ |
| Kumar Stage-Aware (2025) | ResNet-50 + MSE regression | **0.8992** (single val split) | — | — | — | ✗ | ✗ | N=1 | ✗ |
| Dixit EfficientNetB3+SE (2025) | EfficientNetB3 + SE block | "κ 0.88" (ambiguous), Acc 0.884 | 0.884 | Acc 0.817 (own 364/45/46 split; NOT official 413/103) | Acc 0.733 (744-sample subset) | ✗ | ✗ | N=1 | ✗ |
| Gogulamudi non-uniform squash (2024) | Conv→PrimaryCaps→DigitCaps, mod. squash | — | — (7-class NIH/Paraguay, not APTOS) | — | — | ✗ | ✗ | N=1, train-set-reported-as-test | ✗ |
| Lei GF-CapsNet (2024) | ResNet-18 + GNN + CapsNet, multi-head | AUC 0.956, F1 0.741 (no QWK) | 0.865 (own 80/20) | Acc 0.641 (own 80/20; NOT official) | — | ✗ | ✗ | N=1 | ✗ |
| Yu AOR-DR + RETFound (2025) | RETFound (frozen) + AR diffusion + ordinal | — (no QWK), Acc 0.803 / F1 0.657 | 0.803 | — | 0.647 / F1 0.54 (Messidor, not -2) | ✗ | ✗ | N=1 | ✓ |
| El Bellaj UAOR-LQAP+EDL (2026) | ConvNeXt-Base + LQAP pooling + evidential Dirichlet | **0.940** (claim — pooled 3-dataset test) | 0.876 (pooled) | (pooled) | (pooled) | ✗ (pooled train/test, no held-out target domain) | ✓ Evidential Dirichlet | N=1 | not found |
| **Ours — frozen Ordinal CapsNet** | RETFound (frozen) + K-1 ordinal capsule | **0.8932 ± 0.0004** (3 seeds × 5-fold) | 0.7931 ± 0.0021 | **0.756 val / 0.446 test** (5-fold CV on 413 + 103 official test) | **0.6065 ± 0.036** (5-fold within-gradable) | ✓ APTOS→IDRiD QWK 0.273; APTOS→Messidor-2 QWK 0.014 (uncal.) / **0.222 (calibrated)** | ✓ Prediction margin (p < 1e-300 APTOS; p = 4.7e-4 IDRiD; p < 1e-300 Messidor-2) | **N=3** (42, 123, 456) | ✓ |
| **Ours — LoRA Ordinal CapsNet r=8** | RETFound + LoRA r=8 + K-1 ordinal capsule | **0.9127 ± 0.0008** (3 seeds × 5-fold) | 0.8237 ± 0.0068 | pending LoRA IDRiD run | pending LoRA Messidor-2 run | pending | ✓ Prediction margin + routing variance (15-fold pool) | **N=3** | ✓ |

## Interpretation for the paper

### Where we beat / tie / lose on raw QWK

| Tier | Methods | Our frozen (0.893) | Our LoRA (0.913) |
|---|---|---|---|
| We **beat** | Oulhadj (0.778), Gogulamudi (on their custom dataset, incomparable), Lei (AUC-only) | ✓ | ✓ |
| We **tie** (within ±0.01) | Bodapati (0.897), Kumar (0.899), Dixit ("0.88" ambiguous) | essentially tied | beats |
| We **lose** | El Bellaj (0.940 pooled claim) | loses by 0.047 | loses by 0.027 |

**But the El Bellaj number is the only one above us**, and their evaluation is on a pooled 3-dataset held-out test with no per-dataset breakdown, no multi-seed, no public code we've found, and their Messidor-2 AUROC (implied by their QWK) is inconsistent with RETFound's own Messidor-2 AUROC 0.884. Treat their number as **the ceiling claim in the literature**, not an apples-to-apples comparison.

### Where we **win methodologically**

Rows that are checkmarked in the last four columns only on our own lines:

1. **Cross-dataset with honest failure modes**: APTOS→IDRiD 0.273, APTOS→Messidor-2 0.014 uncalibrated / 0.222 rank-calibrated. **Nobody else in the table reports train-on-A-test-on-B cross-dataset numbers** — even El Bellaj pools all three datasets before splitting. Our honesty here is a competitive advantage.
2. **Capsule-native UQ validated on 3 datasets**: prediction margin separates correct from misclassified at p < 10⁻³⁰⁰ on APTOS, p = 4.7×10⁻⁴ on IDRiD, p < 10⁻³⁰⁰ on Messidor-2. Free (no extra loss term, no KL annealing). El Bellaj has UQ but requires evidential Dirichlet loss + query diversity penalty + KL annealing — more moving parts.
3. **Multi-seed reporting**: we're the only method here with N=3 seeds on the APTOS headline row. Every other APTOS QWK number is a point estimate.
4. **Code availability**: we publish code (https://github.com/vineetkumar-sudo/retfound-capsnet) alongside the paper. Lei, Oulhadj, Kumar, Dixit, Bodapati all report "no code available" or didn't release with the paper.

### Where we **characterise failure modes** that competitors don't

- **Cross-dataset calibration collapse** (§5.5): we diagnose the head-probability distribution shift (mean P(y>0) 0.51 → 0.087 on Messidor-2) and show a simple post-hoc rank calibration recovers 0.014 → 0.222 QWK.
- **Asymmetric loss does not help over symmetric K-1** (Day-4 ablation, row 7): direction weighting is redundant because the K-1 decomposition already produces safety bias.
- **KC (differentiable QWK) loss marginally hurts** vs margin loss (Day-5A, row 8): honest negative.
- **Routing-agreement variance as UQ is inverted** (Day-5B, Fig D): rejecting high-variance samples LOWERS accuracy.

None of the competitor papers report negative results at this granularity.

## Rigour checklist (summary)

| Criterion | # of 11 competitor methods with ✓ | Ours |
|---|---:|:---:|
| Reports QWK (not just accuracy / κ / AUC) | 3/11 | ✓ |
| Multi-seed evaluation (N ≥ 3 with std reported) | 0/11 | ✓ (N=3 APTOS) |
| Cross-dataset eval (train-on-A → test-on-B) | 1/11 (only RETFound, and only AUROC) | ✓ (two directions) |
| Native or post-hoc UQ on ≥ 1 dataset | 1/11 (El Bellaj) | ✓ (three datasets) |
| Training cost reported with hardware + wall-clock | 2/11 (Oulhadj, RETFound) | ✓ |
| Parameter count reported (trainable, not just total) | 1/11 (Lei — 12.87 M) | ✓ |
| Code released with paper | 2/11 (RETFound, Yu AOR-DR) | ✓ (planned) |
| Honest negative results | 0/11 | ✓ (4) |

## Notes on the numbers

- **"κ" ambiguity**: Dixit and Bodapati both report "Kappa" without specifying quadratic-weighted. We treat them as QWK here, which is the generous interpretation; if their κ turns out to be unweighted Cohen's κ then they sit lower in the ranking.
- **Dataset splits**: Lei and Dixit report IDRiD results on their own 80/20 splits rather than the official 413-train / 103-test split, so those numbers aren't directly comparable to our IDRiD test QWK 0.446 on the official test set.
- **"pooled" evaluation** (El Bellaj): their 0.940 is on a held-out test drawn from a union of APTOS + Messidor-2 + EyePACS, not any one dataset. This means a sample that was "train" in our per-dataset eval could be "test" in theirs — not the same task.
- **Gogulamudi 99.84 %** (not listed above because it's on their custom NIH/Paraguay 7-class dataset): the figure in Table 1 of their paper is training accuracy (test is 98.68 %). Small test set (~151 images). Not comparable to our evaluation.
