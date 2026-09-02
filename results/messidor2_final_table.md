# Messidor-2 Disease Grading — Results

## Within-Messidor-2 (5-fold CV on 1,744 gradable images)

| # | Model | QWK | Accuracy | Macro F1 | MAE |
|---|-------|-----|----------|----------|-----|
| 1 | MLP + CE | 0.5186 ± 0.0385 | **0.6256 ± 0.0220** | 0.3626 ± 0.0455 | 0.572 ± 0.040 |
| 2 | MLP + MSE | 0.5381 ± 0.0293 | 0.4839 ± 0.0331 | 0.3667 ± 0.0215 | 0.600 ± 0.022 |
| 3 | Vanilla CapsNet | 0.5001 ± 0.0648 | 0.6095 ± 0.0255 | 0.3272 ± 0.0433 | 0.577 ± 0.030 |
| 4 | Ordinal CapsNet | **0.6021 ± 0.0454** | 0.5774 ± 0.0296 | **0.4410 ± 0.0247** | **0.545 ± 0.052** |

## APTOS → Messidor-2 (cross-dataset, no Messidor-2 training)

| # | Model | QWK | Accuracy | Macro F1 | MAE | Binary AUC |
|---|-------|-----|----------|----------|-----|------------|
| 1 | MLP + CE | 0.0017 ± 0.0008 | **0.5837** | **0.1485** | 0.761 | 0.5852 |
| 2 | MLP + MSE | 0.0051 ± 0.0039 | 0.5826 | 0.1474 | 0.761 | 0.5655 |
| 3 | Vanilla CapsNet | 0.0010 ± 0.0034 | 0.5826 | 0.1473 | 0.763 | 0.5702 |
| 4 | Ordinal CapsNet | **0.0172 ± 0.0185** | 0.5826 | 0.1476 | **0.757** | **0.6202** |

Best value per column in **bold** (within each section independently). QWK / Accuracy / Macro F1 / AUC: higher is better; MAE: lower is better. Cross-dataset std columns are omitted because a single ensemble prediction is reported; per-fold stds are tracked in the canonical CSV.