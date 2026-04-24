# Extended calibration: Temperature Scaling + Brier + NLL

All rows pool per-sample predictions across every fold / seed for that config. Temperature T is fitted on a 50% random calibration split (minimising NLL via golden-section search on log T); ECE / Brier / NLL are reported pre- and post-TS on the held-out 50%. ECE uses 10 equal-mass bins on P(y = predicted_grade). Lower is better for every column. T $>$ 1 means the head was over-confident on the cal split; T $<$ 1 means under-confident.

| Dataset | Model | T | ECE pre | ECE post | Brier pre | Brier post | NLL pre | NLL post |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| APTOS | RETFound x Ordinal CapsNet | 0.566 | 0.152 | 0.016 | 0.323 | 0.286 | 0.661 | 0.568 |
| APTOS | RETFound x MLP + K-1 sigmoid | 0.680 | 0.090 | 0.025 | 0.306 | 0.289 | 0.607 | 0.577 |
| APTOS | DINOv2 x Ordinal CapsNet | 0.628 | 0.119 | 0.020 | 0.283 | 0.262 | 0.589 | 0.522 |
| APTOS | DINOv2 x MLP + K-1 sigmoid | 0.600 | 0.099 | 0.016 | 0.284 | 0.265 | 0.558 | 0.500 |
| APTOS | RETFound x Ordinal CapsNet + LoRA | 0.609 | 0.142 | 0.020 | 0.281 | 0.253 | 0.599 | 0.511 |
| Messidor-2 | RETFound x Ordinal CapsNet | 0.718 | 0.091 | 0.054 | 0.542 | 0.527 | 1.033 | 0.999 |
| Messidor-2 | RETFound x MLP + K-1 sigmoid | 0.687 | 0.096 | 0.030 | 0.531 | 0.514 | 1.016 | 0.981 |
| Messidor-2 | DINOv2 x Ordinal CapsNet | 0.607 | 0.134 | 0.047 | 0.474 | 0.444 | 0.892 | 0.840 |
| Messidor-2 | DINOv2 x MLP + K-1 sigmoid | 0.719 | 0.097 | 0.032 | 0.423 | 0.411 | 0.785 | 0.756 |
