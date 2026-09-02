# Extended calibration: Temperature Scaling + Brier + NLL

All rows pool per-sample predictions across every fold / seed for that config. Temperature T is fitted on a 50% random calibration split (minimising NLL via golden-section search on log T); ECE / Brier / NLL are reported pre- and post-TS on the held-out 50%. ECE uses 10 equal-mass bins on P(y = predicted_grade). Lower is better for every column. T $>$ 1 means the head was over-confident on the cal split; T $<$ 1 means under-confident.

| Dataset | Model | T | ECE pre | ECE post | Brier pre | Brier post | NLL pre | NLL post |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| APTOS | RETFound x Ordinal CapsNet | 0.577 | 0.137 | 0.012 | 0.318 | 0.286 | 0.657 | 0.573 |
| APTOS | RETFound x MLP + K-1 sigmoid | 0.663 | 0.094 | 0.014 | 0.305 | 0.286 | 0.608 | 0.570 |
| APTOS | DINOv2 x Ordinal CapsNet | 0.598 | 0.120 | 0.016 | 0.291 | 0.268 | 0.597 | 0.523 |
| APTOS | DINOv2 x MLP + K-1 sigmoid | 0.614 | 0.102 | 0.011 | 0.282 | 0.263 | 0.558 | 0.503 |
| APTOS | RETFound x Ordinal CapsNet + LoRA | 0.624 | 0.132 | 0.017 | 0.277 | 0.254 | 0.587 | 0.509 |
| Messidor-2 | RETFound x Ordinal CapsNet | 0.670 | 0.109 | 0.048 | 0.539 | 0.522 | 1.026 | 0.989 |
| Messidor-2 | RETFound x MLP + K-1 sigmoid | 0.748 | 0.081 | 0.028 | 0.527 | 0.516 | 1.016 | 0.995 |
| Messidor-2 | DINOv2 x Ordinal CapsNet | 0.655 | 0.120 | 0.037 | 0.453 | 0.432 | 0.852 | 0.808 |
| Messidor-2 | DINOv2 x MLP + K-1 sigmoid | 0.769 | 0.076 | 0.022 | 0.430 | 0.420 | 0.804 | 0.781 |
