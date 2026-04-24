# Paired Wilcoxon signed-rank tests

Each row is a paired test on per-fold (seed x fold) QWK values. APTOS pairs pool 3 seeds x 5 folds = 15 paired comparisons; Messidor-2 pairs use 5 folds. Data splits are fixed across seeds via `cfg.data.seed`, so folds are paired between configs and the Wilcoxon assumption of exchangeable pairs under H0 is met.

| Family | A | B | n | mean A | mean B | mean $\Delta$ | A wins | p (two-sided) |
|---|---|---|---:|---:|---:|---:|---:|---:|
| APTOS  head effect (RETFound) | RETFound x Ordinal CapsNet | RETFound x MLP + K-1 sigmoid | 15 | 0.8932 | 0.8866 | +0.0065 | 13/15 | 0.0020 |
| APTOS  head effect (DINOv2) | DINOv2 x Ordinal CapsNet | DINOv2 x MLP + K-1 sigmoid | 15 | 0.9089 | 0.9034 | +0.0055 | 15/15 | 0.0001 |
| APTOS  backbone effect (Ordinal CapsNet head) | DINOv2 x Ordinal CapsNet | RETFound x Ordinal CapsNet | 15 | 0.9089 | 0.8932 | +0.0158 | 13/15 | 0.0004 |
| APTOS  backbone effect (MLP-K1 head) | DINOv2 x MLP + K-1 sigmoid | RETFound x MLP + K-1 sigmoid | 15 | 0.9034 | 0.8866 | +0.0168 | 12/15 | 0.0009 |
| APTOS  LoRA effect (RETFound + CapsNet) | RETFound x Ordinal CapsNet + LoRA | RETFound x Ordinal CapsNet | 15 | 0.9127 | 0.8932 | +0.0196 | 15/15 | 0.0001 |
| Messidor-2  head effect (RETFound) | RETFound x Ordinal CapsNet | RETFound x MLP + K-1 sigmoid | 5 | 0.6065 | 0.5852 | +0.0213 | 5/5 | 0.0625 |
| Messidor-2  head effect (DINOv2) | DINOv2 x Ordinal CapsNet | DINOv2 x MLP + K-1 sigmoid | 5 | 0.7525 | 0.7643 | -0.0118 | 1/5 | 0.1250 |
| Messidor-2  backbone effect (Ordinal CapsNet head) | DINOv2 x Ordinal CapsNet | RETFound x Ordinal CapsNet | 5 | 0.7525 | 0.6065 | +0.1460 | 5/5 | 0.0625 |
| Messidor-2  backbone effect (MLP-K1 head) | DINOv2 x MLP + K-1 sigmoid | RETFound x MLP + K-1 sigmoid | 5 | 0.7643 | 0.5852 | +0.1791 | 5/5 | 0.0625 |

Interpretation: p < 0.05 indicates the observed delta is unlikely under H0 ``both configs have the same median per-fold QWK''. A wins counts folds where QWK(A) > QWK(B).
