# APTOS 2019 — Ablation Table

| # | Model | QWK | Accuracy | Macro F1 | MAE | Regime |
|---|-------|-----|----------|----------|-----|--------|
| 1 | Linear Probe + CE | 0.8389 ± 0.0122 | 0.7971 ± 0.0038 | 0.6011 ± 0.0144 | 0.291 ± 0.011 | 3 seeds x 5-fold |
| 2 | MLP + CE | 0.8755 ± 0.0014 | 0.8091 ± 0.0021 | 0.6325 ± 0.0067 | 0.258 ± 0.002 | 3 seeds x 5-fold |
| 3 | MLP + MSE (Ordinal) | 0.8813 ± 0.0012 | 0.7438 ± 0.0113 | 0.5433 ± 0.0051 | 0.295 ± 0.011 | 3 seeds x 5-fold |
| 4 | MLP + Weighted CE | 0.8638 ± 0.0021 | 0.7796 ± 0.0048 | 0.6262 ± 0.0074 | 0.291 ± 0.006 | 3 seeds x 5-fold |
| 5 | Vanilla CapsNet | 0.8696 ± 0.0026 | 0.8061 ± 0.0046 | 0.6301 ± 0.0044 | 0.263 ± 0.003 | 3 seeds x 5-fold |
| 6 | Ordinal CapsNet | 0.8923 ± 0.0002 | 0.7935 ± 0.0075 | 0.6259 ± 0.0089 | 0.256 ± 0.006 | 3 seeds x 5-fold |
| 7 | + Asymmetric loss | 0.8923 ± 0.0014 | 0.7911 ± 0.0025 | 0.6201 ± 0.0019 | — | 3 seeds x 5-fold |
| 8 | + KC Loss (gamma=0.3) | 0.8916 ± 0.0005 | 0.7840 ± 0.0027 | 0.6126 ± 0.0045 | 0.261 ± 0.002 | 3 seeds x 5-fold |
| 9 | Ordinal CapsNet + LoRA | **0.9143 ± 0.0006** | **0.8240 ± 0.0035** | **0.6701 ± 0.0107** | **0.212 ± 0.003** | 3 seeds x 5-fold, LoRA r=8 |
| 10 | + Non-uniform squash | 0.8890 ± 0.0001 | 0.7893 ± 0.0014 | 0.6122 ± 0.0034 | 0.262 ± 0.001 | 3 seeds x 5-fold |
| 11 | MLP + K-1 sigmoid | 0.8849 ± 0.0015 | 0.7976 ± 0.0088 | 0.6235 ± 0.0106 | 0.258 ± 0.010 | 3 seeds x 5-fold |

Best value per column in **bold**. QWK / Accuracy / Macro F1: higher is better; MAE: lower is better.