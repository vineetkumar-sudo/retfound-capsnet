# Calibration + selective-prediction analysis

All rows pool per-sample predictions across every fold and (where available) seed under an experiment directory. ECE / MCE use 10 equal-mass bins over max chain-rule probability. AURC integrates risk over coverage when ranking samples by confidence (lower is better); Excess-AURC subtracts the oracle AURC for the same error count, making the number comparable across datasets with different base error rates. Two confidence signals are compared: max chain-rule probability (standard) and prediction margin (our capsule-native UQ signal).

| Dataset | Model | Acc | ECE | MCE | AURC(maxp) | Excess-AURC(maxp) | Risk@80(maxp) | Risk@50(maxp) | AURC(margin) | Excess-AURC(margin) | Renorm% |
|---|---|---|---|---|---|---|---|---|---|---|---|
| APTOS | RETFound x Ordinal CapsNet | 0.7935 | 0.1428 | 0.2247 | 0.0591 | 0.0361 | 0.1230 | 0.0328 | 0.0596 | 0.0367 | 100.00 |
| APTOS | RETFound x MLP + K-1 sigmoid | 0.7976 | 0.0978 | 0.1746 | 0.0617 | 0.0397 | 0.1230 | 0.0399 | 0.0601 | 0.0380 | 86.05 |
| APTOS | DINOv2  x Ordinal CapsNet | 0.8037 | 0.1203 | 0.1906 | 0.0497 | 0.0291 | 0.1127 | 0.0158 | 0.0495 | 0.0289 | 100.00 |
| APTOS | DINOv2  x MLP + K-1 sigmoid | 0.8086 | 0.1025 | 0.1694 | 0.0500 | 0.0303 | 0.1082 | 0.0198 | 0.0485 | 0.0289 | 100.00 |
| APTOS | RETFound x Ordinal CapsNet + LoRA | 0.8240 | 0.1310 | 0.1811 | 0.0485 | 0.0320 | 0.1012 | 0.0194 | 0.0477 | 0.0312 | 100.00 |
| Messidor-2 | RETFound x Ordinal CapsNet | 0.5774 | 0.1026 | 0.1881 | 0.2380 | 0.1324 | 0.3448 | 0.2259 | 0.2394 | 0.1338 | 100.00 |
| Messidor-2 | RETFound x MLP + K-1 sigmoid | 0.5837 | 0.0831 | 0.1545 | 0.2353 | 0.1331 | 0.3333 | 0.2328 | 0.2360 | 0.1338 | 100.00 |
| Messidor-2 | DINOv2  x Ordinal CapsNet | 0.6898 | 0.1208 | 0.1939 | 0.1795 | 0.1254 | 0.2416 | 0.1755 | 0.1786 | 0.1245 | 100.00 |
| Messidor-2 | DINOv2  x MLP + K-1 sigmoid | 0.6846 | 0.0775 | 0.1318 | 0.1882 | 0.1321 | 0.2480 | 0.1766 | 0.1839 | 0.1279 | 99.94 |
