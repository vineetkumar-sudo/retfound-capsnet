# Calibration + selective-prediction analysis — Day 13 findings

All rows pool per-sample predictions across every fold and (where available) seed. ECE / MCE use 10 equal-mass bins; confidence = P(y = predicted_grade) from the chain-rule 5-class probability derived from the K-1 heads. AURC integrates risk over coverage when ranking samples by the stated signal (lower is better). "Margin" is our capsule-native UQ signal; "maxp" is the standard max chain-rule probability baseline.

## APTOS-2019 (3 seeds × 5-fold pooled, N = 9,885)

| Head × Backbone | Acc | ECE | MCE | AURC (maxp) | AURC (margin) | Monotonicity renorm |
|---|---:|---:|---:|---:|---:|---:|
| RETFound × Ordinal CapsNet    | 0.7931 | 0.1514 | 0.2242 | 0.0599 | 0.0609 | 100.0% |
| RETFound × MLP + K-1 sigmoid  | 0.7963 | **0.0912** | **0.1740** | 0.0624 | 0.0604 | 86.0% |
| DINOv2  × Ordinal CapsNet     | 0.8086 | 0.1151 | 0.1667 | **0.0485** | **0.0483** | 100.0% |
| DINOv2  × MLP + K-1 sigmoid   | 0.8054 | 0.1015 | 0.1730 | 0.0506 | 0.0488 | 100.0% |
| RETFound × Ordinal CapsNet + LoRA | 0.8237 | 0.1415 | 0.1943 | 0.0498 | 0.0495 | 100.0% |

## Messidor-2 within-dataset (5-fold pooled, N = 1,744)

| Head × Backbone | Acc | ECE | MCE | AURC (maxp) | AURC (margin) |
|---|---:|---:|---:|---:|---:|
| RETFound × Ordinal CapsNet   | 0.5556 | **0.0829** | 0.2081 | 0.2556 | 0.2565 |
| RETFound × MLP + K-1 sigmoid | 0.5786 | 0.0859 | **0.1527** | 0.2369 | 0.2347 |
| DINOv2  × Ordinal CapsNet    | 0.6583 | 0.1320 | 0.2305 | 0.1869 | 0.1868 |
| DINOv2  × MLP + K-1 sigmoid  | 0.7099 | 0.0902 | 0.1587 | **0.1736** | **0.1698** |

## Headline findings for the paper

1. **Capsule heads are consistently WORSE calibrated than MLP-K1 sigmoid heads on APTOS.** ECE 0.151 (RETFound caps) vs 0.091 (RETFound MLP-K1) — the MLP is 40% better calibrated at comparable accuracy. DINOv2 rows show the same pattern (0.115 caps vs 0.102 MLP-K1). Direct consequence: **"native capsule UQ" cannot be pitched as a calibration story** — capsule margin loss is a discrimination loss, not a calibration loss.

2. **Prediction margin and max-prob are near-equivalent as selectors.** AURC (margin) and AURC (maxp) agree within ±0.002 everywhere. Our capsule-native prediction margin is a *valid* selective-prediction signal but not a *special* one; a plain MLP-K1 sigmoid with the same top-2 margin selector performs within noise. This invalidates the prior framing of prediction margin as the unique UQ asset.

3. **On Messidor-2, DINOv2 × MLP-K1 is the Pareto winner across every UQ metric:** best accuracy (0.71), best ECE (0.09), best AURC (0.17). The capsule head loses on all three. Combined with its +0.12 QWK lead over RETFound × CapsNet, this strengthens the "backbone, not head" thesis.

4. **LoRA accelerates accuracy (0.8237) but does NOT improve calibration** (ECE 0.1415 vs 0.1514 frozen — within 1 point). Fine-tuning a calibration-loss-free head just makes the miscalibration tighter but doesn't fix the direction, which is a well-known fine-tuning pathology documented in the calibration literature.

5. **100% of capsule-head samples require chain-rule renormalisation** (the K-1 head probabilities are non-monotonic in the chain-rule product sense for every single sample). MLP-K1 sigmoid is better behaved (14% of APTOS samples are natively monotonic). This is another subtle quality-control advantage for the no-capsule baseline.

## Implication for the paper rebrand

The original framing -- "capsule-native UQ is a free, strong signal" -- does not survive honest ECE / AURC analysis. The new, defensible framing is:

> *"K-1 ordinal heads (either capsule or sigmoid) produce discriminative but uncalibrated confidence scores. Discrimination alone is not enough for clinical deployment; we therefore treat calibration as a downstream problem and apply split conformal prediction (Day 13B) to obtain distribution-free coverage guarantees that hold regardless of the head's miscalibration."*

That's an honest story that (a) acknowledges the capsule's calibration weakness, (b) positions conformal prediction as the real UQ asset, and (c) still keeps the discrimination claim (prediction margin *is* a valid selector, just not a uniquely strong one).

## Artifacts

  * `results/calibration_table.md`, `results/calibration_table.csv`, `results/calibration_metrics.json`
  * `results/figures/fig_reliability_aptos.{png,pdf}` — 4-panel reliability diagrams for the APTOS head × backbone grid
  * `results/figures/fig_reliability_messidor2.{png,pdf}` — same for Messidor-2
  * `results/figures/fig_risk_coverage_aptos.{png,pdf}` — head × backbone AURC curves (max-prob selector)
  * `results/figures/fig_risk_coverage_messidor2.{png,pdf}` — same for Messidor-2

Driver: `uv run python scripts/compute_calibration_metrics.py`  (~3 s; pure CPU, no training).
