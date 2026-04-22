# Messidor-2 Disease Grading — Results

## Within-Messidor-2 (5-fold CV on 1,744 gradable images)

| # | Model | QWK | Accuracy | Macro F1 | MAE |
|---|-------|-----|----------|----------|-----|
| 1 | MLP + CE | 0.5344 ± 0.0211 | **0.6170 ± 0.0215** | 0.4131 ± 0.0140 | 0.573 ± 0.021 |
| 2 | MLP + MSE | 0.5566 ± 0.0278 | 0.4822 ± 0.0296 | 0.3849 ± 0.0123 | 0.596 ± 0.033 |
| 3 | Vanilla CapsNet | 0.5001 ± 0.0648 | 0.6095 ± 0.0255 | 0.3272 ± 0.0433 | 0.577 ± 0.030 |
| 4 | Ordinal CapsNet | **0.6065 ± 0.0363** | 0.5556 ± 0.0451 | **0.4426 ± 0.0293** | **0.558 ± 0.052** |

## APTOS → Messidor-2 (cross-dataset, no Messidor-2 training)

| # | Model | QWK | Accuracy | Macro F1 | MAE | Binary AUC |
|---|-------|-----|----------|----------|-----|------------|
| 1 | MLP + CE | 0.0000 ± 0.0015 | **0.5831** | 0.1473 | 0.762 | 0.5795 |
| 2 | MLP + MSE | 0.0037 ± 0.0184 | 0.5826 | 0.1474 | 0.762 | 0.5874 |
| 3 | Vanilla CapsNet | 0.0010 ± 0.0034 | 0.5826 | 0.1473 | 0.763 | 0.5702 |
| 4 | Ordinal CapsNet | **0.0086 ± 0.0136** | 0.5831 | **0.1475** | **0.760** | **0.6107** |

Best value per column in **bold** (within each section independently). QWK / Accuracy / Macro F1 / AUC: higher is better; MAE: lower is better. Cross-dataset std columns are omitted because a single ensemble prediction is reported; per-fold stds are tracked in the canonical CSV.