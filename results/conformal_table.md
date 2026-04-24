# Split conformal prediction — paper-ready table

Five random cal/test 50:50 splits per config; mean ± std reported. Each row is pooled across all folds × seeds for that config (same pool as `compute_calibration_metrics.py`). Empty cells where no preds exist.

## Coverage and set-size at α = 0.10 (target 90% coverage)

Four nonconformity scores compared: LAC (Sadinle 2019), APS (Romano 2020), RAPS (Angelopoulos 2021, $k_\text{reg}{=}1$, $\lambda{=}0.01$), and CCP (Mondrian per-class quantile, Vovk 2003). LAC/APS/RAPS are all marginal-coverage methods; CCP is class-conditional. $|S|$ is mean prediction set size (smaller at equal coverage is better).

| Dataset | Model | LAC cov | LAC \|S\| | APS cov | APS \|S\| | RAPS cov | RAPS \|S\| | CCP cov | CCP \|S\| |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| APTOS | RETFound x Ordinal CapsNet | 0.902 | 1.38 | 0.939 | 1.80 | 0.936 | 1.77 | 0.903 | 1.86 |
| APTOS | RETFound x MLP + K-1 sigmoid | 0.904 | 1.38 | 0.940 | 1.89 | 0.937 | 1.85 | 0.898 | 1.78 |
| APTOS | DINOv2  x Ordinal CapsNet | 0.903 | 1.33 | 0.948 | 1.73 | 0.944 | 1.70 | 0.909 | 1.78 |
| APTOS | DINOv2  x MLP + K-1 sigmoid | 0.903 | 1.30 | 0.948 | 1.79 | 0.945 | 1.75 | 0.914 | 1.65 |
| APTOS | RETFound x Ordinal CapsNet + LoRA | 0.904 | 1.27 | 0.938 | 1.69 | 0.935 | 1.66 | 0.905 | 1.80 |
| Messidor-2 | RETFound x Ordinal CapsNet | 0.903 | 2.30 | 0.903 | 2.54 | 0.904 | 2.54 | 0.901 | 3.48 |
| Messidor-2 | RETFound x MLP + K-1 sigmoid | 0.905 | 2.32 | 0.904 | 2.50 | 0.903 | 2.52 | 0.904 | 3.25 |
| Messidor-2 | DINOv2  x Ordinal CapsNet | 0.905 | 1.94 | 0.894 | 2.06 | 0.896 | 2.07 | 0.908 | 2.95 |
| Messidor-2 | DINOv2  x MLP + K-1 sigmoid | 0.900 | 1.75 | 0.903 | 1.97 | 0.902 | 1.97 | 0.905 | 2.38 |

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

## Class-conditional coverage at α = 0.10 (single-split seed 42)

Worst-per-class coverage is the honest deployment metric: the marginal guarantee from LAC/APS/RAPS does not transfer to per-class coverage. CCP (Mondrian, per-class quantile) is the canonical fix and is the closest non-RC3P baseline achievable on MPS without new training. Values are per-class coverage; `min` is the worst class; `|S|` is the mean CCP set size at this alpha.

### APS (single global threshold)

| Dataset | Model | C0 | C1 | C2 | C3 | C4 | min |
|---|---|---:|---:|---:|---:|---:|---:|
| APTOS | RETFound x Ordinal CapsNet | 0.985 | 0.840 | 0.951 | 0.890 | 0.779 | 0.779 |
| APTOS | RETFound x MLP + K-1 sigmoid | 0.990 | 0.838 | 0.944 | 0.862 | 0.802 | 0.802 |
| APTOS | DINOv2  x Ordinal CapsNet | 0.993 | 0.838 | 0.962 | 0.905 | 0.837 | 0.837 |
| APTOS | DINOv2  x MLP + K-1 sigmoid | 0.995 | 0.863 | 0.953 | 0.894 | 0.845 | 0.845 |
| APTOS | RETFound x Ordinal CapsNet + LoRA | 0.989 | 0.782 | 0.947 | 0.951 | 0.807 | 0.782 |
| Messidor-2 | RETFound x Ordinal CapsNet | 0.929 | 0.939 | 0.778 | 0.744 | 0.833 | 0.744 |
| Messidor-2 | RETFound x MLP + K-1 sigmoid | 0.933 | 0.947 | 0.806 | 0.721 | 0.833 | 0.721 |
| Messidor-2 | DINOv2  x Ordinal CapsNet | 0.966 | 0.863 | 0.667 | 0.977 | 0.833 | 0.667 |
| Messidor-2 | DINOv2  x MLP + K-1 sigmoid | 0.976 | 0.870 | 0.711 | 0.884 | 0.583 | 0.583 |

### CCP (Mondrian, per-class quantile)

| Dataset | Model | C0 | C1 | C2 | C3 | C4 | min | CCP \|S\| |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| APTOS | RETFound x Ordinal CapsNet | 0.909 | 0.901 | 0.907 | 0.915 | 0.908 | 0.901 | 1.86 |
| APTOS | RETFound x MLP + K-1 sigmoid | 0.894 | 0.910 | 0.905 | 0.926 | 0.901 | 0.894 | 1.78 |
| APTOS | DINOv2  x Ordinal CapsNet | 0.917 | 0.874 | 0.898 | 0.929 | 0.908 | 0.874 | 1.78 |
| APTOS | DINOv2  x MLP + K-1 sigmoid | 0.932 | 0.882 | 0.887 | 0.887 | 0.901 | 0.882 | 1.65 |
| APTOS | RETFound x Ordinal CapsNet + LoRA | 0.911 | 0.912 | 0.901 | 0.933 | 0.931 | 0.901 | 1.80 |
| Messidor-2 | RETFound x Ordinal CapsNet | 0.903 | 0.985 | 0.883 | 0.977 | 0.917 | 0.883 | 3.48 |
| Messidor-2 | RETFound x MLP + K-1 sigmoid | 0.911 | 0.924 | 0.894 | 0.884 | 0.833 | 0.833 | 3.25 |
| Messidor-2 | DINOv2  x Ordinal CapsNet | 0.917 | 0.939 | 0.822 | 0.953 | 0.917 | 0.822 | 2.95 |
| Messidor-2 | DINOv2  x MLP + K-1 sigmoid | 0.929 | 0.893 | 0.839 | 0.977 | 0.917 | 0.839 | 2.38 |
