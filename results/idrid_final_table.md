# IDRiD Disease Grading — Ablation Table

## Within-IDRiD (5-fold CV on 413 train → official 103 test set)

| # | Model | Val QWK | Test QWK | Test Accuracy | Test Macro F1 | Test MAE |
|---|-------|---------|----------|---------------|---------------|----------|
| 1 | MLP + CE | 0.7453 ± 0.0354 | 0.4122 ± 0.0258 | **0.4311 ± 0.0311** | 0.3120 ± 0.0261 | 0.986 ± 0.059 |
| 2 | MLP + MSE | 0.7494 ± 0.0395 | **0.4652 ± 0.0287** | 0.3107 ± 0.0261 | 0.2623 ± 0.0317 | **0.946 ± 0.029** |
| 3 | Vanilla CapsNet | 0.7484 ± 0.0444 | 0.4124 ± 0.0583 | 0.4272 ± 0.0275 | **0.3230 ± 0.0166** | 0.979 ± 0.063 |
| 4 | Ordinal CapsNet | 0.7557 ± 0.0448 | 0.4456 ± 0.0361 | 0.3825 ± 0.0258 | 0.3151 ± 0.0308 | 0.979 ± 0.049 |

## Cross-dataset generalization

| Evaluation | Test QWK | Test Accuracy | Test Macro F1 | Test MAE |
|------------|----------|---------------|---------------|----------|
| APTOS → IDRiD  (Ordinal, ensemble) | 0.2730 ± 0.0198 | 0.3495 | 0.2312 | 1.019 |

Best value per column (within-IDRiD rows only) in **bold**. Cross-dataset row uses std across APTOS folds.