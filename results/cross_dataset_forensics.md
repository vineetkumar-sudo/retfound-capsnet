# Cross-dataset collapse forensics (APTOS → Messidor-2, frozen RETFound backbone)

Investigates why the APTOS-trained Ordinal CapsNet yields QWK $\approx 0.01$ on Messidor-2 across every baseline.

## Pooled numbers (5 folds × 1,744 Messidor-2 images = 8,720 predictions)

**Ground-truth Messidor-2 distribution:**

| Grade | Count | Fraction |
|---|---:|---:|
| 0 (No DR)     | 5,085 | 58.3% |
| 1 (Mild)      | 1,350 | 15.5% |
| 2 (Moderate)  | 1,735 | 19.9% |
| 3 (Severe)    |   375 |  4.3% |
| 4 (PDR)       |   175 |  2.0% |

**Predicted distribution under K-1 threshold at 0.5:**

| Grade | Count | Fraction |
|---|---:|---:|
| 0 | 8,672 | **99.4%** |
| 1 |    26 |  0.3% |
| 2 |    12 |  0.1% |
| 3 |    10 |  0.1% |
| 4 |     0 |  0.0% |

**Mean predicted head probabilities under Messidor-2 features:**

| Head | Mean P(y>k) |
|---|---:|
| y > 0 | 0.087 |
| y > 1 | 0.088 |
| y > 2 | 0.066 |
| y > 3 | 0.037 |

All four heads fire *below* the 0.5 threshold on virtually every sample → nearly every prediction collapses to Grade 0, which in turn produces a QWK indistinguishable from chance on an imbalanced target.

## Why this is *not* a bug in our method

1. **Argmax-of-chain-rule is worse, not better**: replacing the K-1 threshold decoding with `argmax` over the chain-rule 5-class probabilities drops the Grade-0 rate from 99.4% to 99.9% and QWK from 0.021 to 0.008. The collapse is not a threshold artefact; it is a feature-distribution artefact.

2. **Accuracy matches the naive-constant-predictor baseline**: a constant-Grade-0 predictor on a 58.3%-Grade-0 target achieves exactly 0.583 accuracy. Our pooled accuracy is 0.5827. We are, operationally, a near-constant-Grade-0 predictor.

3. **Independent replication with the same direction**: Zoellin et al. (ARVO Imaging 2024) report "Frozen RETFound" on Messidor with QKappa $= 0.0$ at linear-probe adaptation (Table 2 of the abstract), while unfrozen RETFound recovers to 0.592. Poyrazer et al. (*Frontiers in Medicine* 2026) report RETFound's frozen representations drop from binary AUC 0.98 on APTOS to 0.697 on external Messidor-2 ($\Delta$AUC $= -0.286$), **falling below an ImageNet baseline**. The frozen-RETFound cross-dataset collapse is a property of the RETFound feature space, not of our head.

4. **Literature numbers in the 0.30–0.68 range for APTOS→Messidor-2 are not comparable** — those are all *fine-tuned* backbones, not frozen features. Zoellin's unfrozen RETFound on Messidor is 0.592, unfrozen DINOv2 0.714; both match the literature range.

## Implication for the paper

The paper's cross-dataset story is now straightforward:

- Frozen features from RETFound collapse on external domain shift. Not a head bug.
- Post-hoc prior-matching (what we label as *rank calibration* in the current draft; properly attributed to Saerens–Latinne–Decaestecker 2002 in the revised draft) partially restores the ordinal ranking signal — QWK 0.009 → 0.222 — but does not reach a clinically useful operating point.
- The revised draft frames this as a **vulnerability diagnostic**, not a **domain-adaptation contribution**. It is a lightweight sanity check demonstrating that the collapse is threshold-based, not representational.
- Any deployment claim is explicitly withdrawn.
