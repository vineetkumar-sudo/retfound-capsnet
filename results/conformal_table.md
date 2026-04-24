# Split conformal prediction — paper-ready table

Five random cal/test 50:50 splits per config; mean ± std reported. Each row is pooled across all folds × seeds for that config (same pool as `compute_calibration_metrics.py`). Empty cells where no preds exist.

## Coverage and set-size at α = 0.10 (target 90% coverage)

| Dataset | Model | LAC cov | LAC mean\|S\| | APS cov | APS mean\|S\| |
|---|---|---:|---:|---:|---:|
| APTOS | RETFound x Ordinal CapsNet | 0.902 ± 0.005 | 1.38 ± 0.02 | 0.939 ± 0.005 | 1.80 ± 0.03 |
| APTOS | RETFound x MLP + K-1 sigmoid | 0.904 ± 0.005 | 1.38 ± 0.02 | 0.940 ± 0.003 | 1.89 ± 0.03 |
| APTOS | DINOv2  x Ordinal CapsNet | 0.903 ± 0.004 | 1.33 ± 0.02 | 0.948 ± 0.004 | 1.73 ± 0.03 |
| APTOS | DINOv2  x MLP + K-1 sigmoid | 0.903 ± 0.004 | 1.30 ± 0.01 | 0.948 ± 0.003 | 1.79 ± 0.01 |
| APTOS | RETFound x Ordinal CapsNet + LoRA | 0.904 ± 0.006 | 1.27 ± 0.01 | 0.938 ± 0.003 | 1.69 ± 0.01 |
| Messidor-2 | RETFound x Ordinal CapsNet | 0.903 ± 0.017 | 2.30 ± 0.07 | 0.903 ± 0.008 | 2.54 ± 0.05 |
| Messidor-2 | RETFound x MLP + K-1 sigmoid | 0.905 ± 0.011 | 2.32 ± 0.04 | 0.904 ± 0.010 | 2.50 ± 0.07 |
| Messidor-2 | DINOv2  x Ordinal CapsNet | 0.905 ± 0.008 | 1.94 ± 0.03 | 0.894 ± 0.005 | 2.06 ± 0.05 |
| Messidor-2 | DINOv2  x MLP + K-1 sigmoid | 0.900 ± 0.004 | 1.75 ± 0.02 | 0.903 ± 0.011 | 1.97 ± 0.06 |

## Coverage and set-size at α = 0.05 (target 95% coverage)

| Dataset | Model | LAC cov | LAC mean\|S\| | APS cov | APS mean\|S\| |
|---|---|---:|---:|---:|---:|
| APTOS | RETFound x Ordinal CapsNet | 0.952 ± 0.005 | 1.81 ± 0.03 | 0.966 ± 0.004 | 2.20 ± 0.04 |
| APTOS | RETFound x MLP + K-1 sigmoid | 0.951 ± 0.003 | 1.78 ± 0.02 | 0.967 ± 0.003 | 2.34 ± 0.05 |
| APTOS | DINOv2  x Ordinal CapsNet | 0.951 ± 0.001 | 1.65 ± 0.01 | 0.971 ± 0.002 | 2.06 ± 0.02 |
| APTOS | DINOv2  x MLP + K-1 sigmoid | 0.950 ± 0.003 | 1.60 ± 0.02 | 0.971 ± 0.001 | 2.15 ± 0.03 |
| APTOS | RETFound x Ordinal CapsNet + LoRA | 0.950 ± 0.004 | 1.72 ± 0.03 | 0.963 ± 0.003 | 2.05 ± 0.03 |
| Messidor-2 | RETFound x Ordinal CapsNet | 0.954 ± 0.008 | 2.86 ± 0.06 | 0.950 ± 0.007 | 3.09 ± 0.04 |
| Messidor-2 | RETFound x MLP + K-1 sigmoid | 0.959 ± 0.008 | 2.81 ± 0.07 | 0.948 ± 0.008 | 2.95 ± 0.10 |
| Messidor-2 | DINOv2  x Ordinal CapsNet | 0.950 ± 0.007 | 2.38 ± 0.05 | 0.945 ± 0.008 | 2.57 ± 0.06 |
| Messidor-2 | DINOv2  x MLP + K-1 sigmoid | 0.948 ± 0.008 | 2.17 ± 0.05 | 0.955 ± 0.012 | 2.47 ± 0.10 |

## Class-conditional coverage at α = 0.10, APS score (single-split seed 42)

Worst-per-class coverage is the honest deployment metric: LAC/APS only guarantee marginal coverage; minority classes can fall below 1-α. Gap = best - worst class coverage within each row.

| Dataset | Model | C0 | C1 | C2 | C3 | C4 | min | max-min |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| APTOS | RETFound x Ordinal CapsNet | 0.985 | 0.840 | 0.951 | 0.890 | 0.779 | 0.779 | 0.207 |
| APTOS | RETFound x MLP + K-1 sigmoid | 0.990 | 0.838 | 0.944 | 0.862 | 0.802 | 0.802 | 0.188 |
| APTOS | DINOv2  x Ordinal CapsNet | 0.993 | 0.838 | 0.962 | 0.905 | 0.837 | 0.837 | 0.156 |
| APTOS | DINOv2  x MLP + K-1 sigmoid | 0.995 | 0.863 | 0.953 | 0.894 | 0.845 | 0.845 | 0.150 |
| APTOS | RETFound x Ordinal CapsNet + LoRA | 0.989 | 0.782 | 0.947 | 0.951 | 0.807 | 0.782 | 0.207 |
| Messidor-2 | RETFound x Ordinal CapsNet | 0.929 | 0.939 | 0.778 | 0.744 | 0.833 | 0.744 | 0.195 |
| Messidor-2 | RETFound x MLP + K-1 sigmoid | 0.933 | 0.947 | 0.806 | 0.721 | 0.833 | 0.721 | 0.226 |
| Messidor-2 | DINOv2  x Ordinal CapsNet | 0.966 | 0.863 | 0.667 | 0.977 | 0.833 | 0.667 | 0.310 |
| Messidor-2 | DINOv2  x MLP + K-1 sigmoid | 0.976 | 0.870 | 0.711 | 0.884 | 0.583 | 0.583 | 0.393 |
