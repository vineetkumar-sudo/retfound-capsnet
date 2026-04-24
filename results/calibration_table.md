# Calibration + selective-prediction analysis

All rows pool per-sample predictions across every fold and (where available) seed under an experiment directory. ECE / MCE use 10 equal-mass bins over max chain-rule probability. AURC integrates risk over coverage when ranking samples by confidence (lower is better); Excess-AURC subtracts the oracle AURC for the same error count, making the number comparable across datasets with different base error rates. Two confidence signals are compared: max chain-rule probability (standard) and prediction margin (our capsule-native UQ signal).

| Dataset | Model | Acc | ECE | MCE | AURC(maxp) | Excess-AURC(maxp) | Risk@80(maxp) | Risk@50(maxp) | AURC(margin) | Excess-AURC(margin) | Renorm% |
|---|---|---|---|---|---|---|---|---|---|---|---|
| APTOS | RETFound x Ordinal CapsNet | 0.7931 | 0.1514 | 0.2242 | 0.0599 | 0.0369 | 0.1232 | 0.0350 | 0.0609 | 0.0378 | 100.00 |
| APTOS | RETFound x MLP + K-1 sigmoid | 0.7963 | 0.0912 | 0.1740 | 0.0624 | 0.0400 | 0.1243 | 0.0431 | 0.0604 | 0.0381 | 86.05 |
| APTOS | DINOv2  x Ordinal CapsNet | 0.8086 | 0.1151 | 0.1667 | 0.0485 | 0.0289 | 0.1103 | 0.0158 | 0.0483 | 0.0287 | 100.00 |
| APTOS | DINOv2  x MLP + K-1 sigmoid | 0.8054 | 0.1015 | 0.1730 | 0.0506 | 0.0303 | 0.1112 | 0.0180 | 0.0488 | 0.0285 | 100.00 |
| APTOS | RETFound x Ordinal CapsNet + LoRA | 0.8237 | 0.1415 | 0.1943 | 0.0498 | 0.0332 | 0.0999 | 0.0231 | 0.0495 | 0.0329 | 100.00 |
| Messidor-2 | RETFound x Ordinal CapsNet | 0.5556 | 0.0829 | 0.2081 | 0.2556 | 0.1376 | 0.3706 | 0.2305 | 0.2565 | 0.1385 | 100.00 |
| Messidor-2 | RETFound x MLP + K-1 sigmoid | 0.5786 | 0.0859 | 0.1527 | 0.2369 | 0.1319 | 0.3462 | 0.2317 | 0.2347 | 0.1298 | 100.00 |
| Messidor-2 | DINOv2  x Ordinal CapsNet | 0.6583 | 0.1320 | 0.2305 | 0.1869 | 0.1204 | 0.2652 | 0.1686 | 0.1868 | 0.1202 | 100.00 |
| Messidor-2 | DINOv2  x MLP + K-1 sigmoid | 0.7099 | 0.0902 | 0.1587 | 0.1736 | 0.1266 | 0.2258 | 0.1594 | 0.1698 | 0.1228 | 99.89 |
