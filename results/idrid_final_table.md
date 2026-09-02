# IDRiD Disease Grading — Ablation Table

## Within-IDRiD (5-fold CV on 413 train → official 103 test set)

| # | Model | Val QWK | Test QWK | Test Accuracy | Test Macro F1 | Test MAE |
|---|-------|---------|----------|---------------|---------------|----------|
| 1 | MLP + CE | 0.7575 ± 0.0351 | 0.4078 ± 0.0584 | **0.4369 ± 0.0368** | 0.3185 ± 0.0361 | 0.977 ± 0.061 |
| 2 | MLP + MSE | 0.7551 ± 0.0324 | **0.4614 ± 0.0289** | 0.3184 ± 0.0241 | 0.2716 ± 0.0285 | **0.934 ± 0.032** |
| 3 | Vanilla CapsNet | 0.7484 ± 0.0444 | 0.4124 ± 0.0583 | 0.4272 ± 0.0275 | **0.3230 ± 0.0166** | 0.979 ± 0.063 |
| 4 | Ordinal CapsNet | 0.7595 ± 0.0461 | 0.4232 ± 0.0192 | 0.3825 ± 0.0117 | 0.3179 ± 0.0156 | 0.977 ± 0.012 |

## Cross-dataset generalization

| Evaluation | Test QWK | Test Accuracy | Test Macro F1 | Test MAE |
|------------|----------|---------------|---------------|----------|
| APTOS → IDRiD  (Ordinal, ensemble) | 0.2788 ± 0.0389 | 0.3398 | 0.2360 | 1.029 |

Best value per column (within-IDRiD rows only) in **bold**. Cross-dataset row uses std across APTOS folds.