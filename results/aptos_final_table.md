# APTOS 2019 — Ablation Table

| # | Model | QWK | Accuracy | Macro F1 | MAE | Regime |
|---|-------|-----|----------|----------|-----|--------|
| 1 | Linear Probe + CE | 0.8389 ± 0.0122 | 0.7971 ± 0.0038 | 0.6011 ± 0.0144 | 0.291 ± 0.011 | 3 seeds x 5-fold |
| 2 | MLP + CE | 0.8783 ± 0.0022 | 0.8057 ± 0.0012 | 0.6271 ± 0.0021 | 0.259 ± 0.001 | 3 seeds x 5-fold |
| 3 | MLP + MSE (Ordinal) | 0.8831 ± 0.0011 | 0.7445 ± 0.0016 | 0.5491 ± 0.0069 | 0.294 ± 0.002 | 3 seeds x 5-fold |
| 4 | MLP + Weighted CE | 0.8607 ± 0.0014 | 0.7679 ± 0.0022 | 0.6156 ± 0.0046 | 0.307 ± 0.002 | 3 seeds x 5-fold |
| 5 | Vanilla CapsNet | 0.8696 ± 0.0026 | 0.8061 ± 0.0046 | 0.6301 ± 0.0044 | 0.263 ± 0.003 | 3 seeds x 5-fold |
| 6 | Ordinal CapsNet | 0.8932 ± 0.0004 | 0.7931 ± 0.0021 | 0.6255 ± 0.0022 | 0.254 ± 0.002 | 3 seeds x 5-fold |
| 7 | + Asymmetric loss | 0.8923 ± 0.0014 | 0.7911 ± 0.0025 | 0.6201 ± 0.0019 | 0.256 ± 0.004 | 3 seeds x 5-fold |
| 8 | + KC Loss (gamma=0.3) | 0.8914 ± 0.0006 | 0.7819 ± 0.0025 | 0.6130 ± 0.0028 | 0.264 ± 0.003 | 3 seeds x 5-fold |
| 9 | Ordinal CapsNet + LoRA | **0.9127 ± 0.0008** | **0.8237 ± 0.0068** | **0.6627 ± 0.0105** | **0.214 ± 0.006** | 3 seeds x 5-fold, LoRA r=8 |
| 10 | + Non-uniform squash | 0.8894 ± 0.0006 | 0.7857 ± 0.0059 | 0.6097 ± 0.0095 | 0.264 ± 0.004 | 3 seeds x 5-fold |
| 11 | MLP + K-1 sigmoid | 0.8866 ± 0.0004 | 0.7963 ± 0.0041 | 0.6250 ± 0.0048 | 0.257 ± 0.003 | 3 seeds x 5-fold |

Best value per column in **bold**. QWK / Accuracy / Macro F1: higher is better; MAE: lower is better.