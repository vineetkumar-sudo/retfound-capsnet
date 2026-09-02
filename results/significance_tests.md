# Paired Wilcoxon signed-rank tests

Each row is a paired test on per-fold (seed x fold) QWK values. APTOS pairs pool 3 seeds x 5 folds = 15 paired comparisons; Messidor-2 pairs use 5 folds. Data splits are fixed across seeds via `cfg.data.seed`, so folds are paired between configs and the Wilcoxon assumption of exchangeable pairs under H0 is met.

| Family | A | B | n | mean A | mean B | mean $\Delta$ | A wins | p (two-sided) |
|---|---|---|---:|---:|---:|---:|---:|---:|
| APTOS  head effect (RETFound) | RETFound x Ordinal CapsNet | RETFound x MLP + K-1 sigmoid | 15 | 0.8923 | 0.8849 | +0.0073 | 14/15 | 0.0009 |
| APTOS  head effect (DINOv2) | DINOv2 x Ordinal CapsNet | DINOv2 x MLP + K-1 sigmoid | 15 | 0.9074 | 0.9041 | +0.0033 | 12/15 | 0.0125 |
| APTOS  backbone effect (Ordinal CapsNet head) | DINOv2 x Ordinal CapsNet | RETFound x Ordinal CapsNet | 15 | 0.9074 | 0.8923 | +0.0152 | 13/15 | 0.0004 |
| APTOS  backbone effect (MLP-K1 head) | DINOv2 x MLP + K-1 sigmoid | RETFound x MLP + K-1 sigmoid | 15 | 0.9041 | 0.8849 | +0.0192 | 14/15 | 0.0003 |
| APTOS  LoRA effect (RETFound + CapsNet) | RETFound x Ordinal CapsNet + LoRA | RETFound x Ordinal CapsNet | 15 | 0.9143 | 0.8923 | +0.0220 | 15/15 | 0.0001 |
| Messidor-2  head effect (RETFound) | RETFound x Ordinal CapsNet | RETFound x MLP + K-1 sigmoid | 5 | 0.6021 | 0.5900 | +0.0121 | 3/5 | 0.6250 |
| Messidor-2  head effect (DINOv2) | DINOv2 x Ordinal CapsNet | DINOv2 x MLP + K-1 sigmoid | 5 | 0.7586 | 0.7587 | -0.0002 | 3/5 | 1.0000 |
| Messidor-2  backbone effect (Ordinal CapsNet head) | DINOv2 x Ordinal CapsNet | RETFound x Ordinal CapsNet | 5 | 0.7586 | 0.6021 | +0.1565 | 5/5 | 0.0625 |
| Messidor-2  backbone effect (MLP-K1 head) | DINOv2 x MLP + K-1 sigmoid | RETFound x MLP + K-1 sigmoid | 5 | 0.7587 | 0.5900 | +0.1687 | 5/5 | 0.0625 |

Interpretation: p < 0.05 indicates the observed delta is unlikely under H0 ``both configs have the same median per-fold QWK''. A wins counts folds where QWK(A) > QWK(B).
