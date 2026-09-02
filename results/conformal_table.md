# Split conformal prediction — paper-ready table

Five random cal/test 50:50 splits per config; mean ± std reported. Each row is pooled across all folds × seeds for that config (same pool as `compute_calibration_metrics.py`). Empty cells where no preds exist.

## Coverage and set-size at α = 0.10 (target 90% coverage)

Four nonconformity scores compared: LAC (Sadinle 2019), APS (Romano 2020), RAPS (Angelopoulos 2021, $k_\text{reg}{=}1$, $\lambda{=}0.01$), and CCP (Mondrian per-class quantile, Vovk 2003). LAC/APS/RAPS are all marginal-coverage methods; CCP is class-conditional. $|S|$ is mean prediction set size (smaller at equal coverage is better).

| Dataset | Model | LAC cov | LAC \|S\| | APS cov | APS \|S\| | RAPS cov | RAPS \|S\| | CCP cov | CCP \|S\| |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| APTOS | RETFound x Ordinal CapsNet | 0.903 | 1.38 | 0.943 | 1.81 | 0.940 | 1.79 | 0.904 | 1.84 |
| APTOS | RETFound x MLP + K-1 sigmoid | 0.902 | 1.37 | 0.939 | 1.87 | 0.936 | 1.82 | 0.897 | 1.84 |
| APTOS | DINOv2  x Ordinal CapsNet | 0.906 | 1.32 | 0.948 | 1.76 | 0.945 | 1.74 | 0.908 | 1.75 |
| APTOS | DINOv2  x MLP + K-1 sigmoid | 0.900 | 1.29 | 0.945 | 1.80 | 0.943 | 1.77 | 0.912 | 1.66 |
| APTOS | RETFound x Ordinal CapsNet + LoRA | 0.903 | 1.28 | 0.942 | 1.69 | 0.939 | 1.66 | 0.913 | 1.67 |
| Messidor-2 | RETFound x Ordinal CapsNet | 0.909 | 2.33 | 0.903 | 2.50 | 0.902 | 2.49 | 0.907 | 3.44 |
| Messidor-2 | RETFound x MLP + K-1 sigmoid | 0.904 | 2.30 | 0.902 | 2.48 | 0.901 | 2.50 | 0.915 | 3.41 |
| Messidor-2 | DINOv2  x Ordinal CapsNet | 0.903 | 1.85 | 0.895 | 2.03 | 0.896 | 2.04 | 0.904 | 2.73 |
| Messidor-2 | DINOv2  x MLP + K-1 sigmoid | 0.903 | 1.83 | 0.904 | 2.03 | 0.905 | 2.03 | 0.908 | 2.54 |

## Coverage and set-size at α = 0.05 (target 95% coverage)

| Dataset | Model | LAC cov | LAC mean\|S\| | APS cov | APS mean\|S\| |
|---|---|---:|---:|---:|---:|
| APTOS | RETFound x Ordinal CapsNet | 0.952 ± 0.003 | 1.80 ± 0.02 | 0.967 ± 0.002 | 2.22 ± 0.05 |
| APTOS | RETFound x MLP + K-1 sigmoid | 0.951 ± 0.003 | 1.82 ± 0.02 | 0.967 ± 0.002 | 2.37 ± 0.03 |
| APTOS | DINOv2  x Ordinal CapsNet | 0.952 ± 0.003 | 1.67 ± 0.02 | 0.972 ± 0.003 | 2.08 ± 0.03 |
| APTOS | DINOv2  x MLP + K-1 sigmoid | 0.950 ± 0.002 | 1.65 ± 0.01 | 0.970 ± 0.002 | 2.20 ± 0.03 |
| APTOS | RETFound x Ordinal CapsNet + LoRA | 0.948 ± 0.005 | 1.65 ± 0.02 | 0.964 ± 0.002 | 1.99 ± 0.03 |
| Messidor-2 | RETFound x Ordinal CapsNet | 0.954 ± 0.009 | 2.78 ± 0.08 | 0.954 ± 0.007 | 2.97 ± 0.06 |
| Messidor-2 | RETFound x MLP + K-1 sigmoid | 0.958 ± 0.008 | 2.91 ± 0.08 | 0.949 ± 0.007 | 2.99 ± 0.07 |
| Messidor-2 | DINOv2  x Ordinal CapsNet | 0.953 ± 0.009 | 2.35 ± 0.07 | 0.950 ± 0.008 | 2.61 ± 0.06 |
| Messidor-2 | DINOv2  x MLP + K-1 sigmoid | 0.952 ± 0.010 | 2.33 ± 0.08 | 0.955 ± 0.006 | 2.56 ± 0.06 |

## Class-conditional coverage at α = 0.10 (single-split seed 42)

Worst-per-class coverage is the honest deployment metric: the marginal guarantee from LAC/APS/RAPS does not transfer to per-class coverage. CCP (Mondrian, per-class quantile) is the canonical fix and is the closest non-RC3P baseline achievable on MPS without new training. Values are per-class coverage; `min` is the worst class; `|S|` is the mean CCP set size at this alpha.

### APS (single global threshold)

| Dataset | Model | C0 | C1 | C2 | C3 | C4 | min |
|---|---|---:|---:|---:|---:|---:|---:|
| APTOS | RETFound x Ordinal CapsNet | 0.987 | 0.859 | 0.948 | 0.883 | 0.817 | 0.817 |
| APTOS | RETFound x MLP + K-1 sigmoid | 0.988 | 0.861 | 0.947 | 0.862 | 0.766 | 0.766 |
| APTOS | DINOv2  x Ordinal CapsNet | 0.992 | 0.836 | 0.960 | 0.912 | 0.845 | 0.836 |
| APTOS | DINOv2  x MLP + K-1 sigmoid | 0.994 | 0.823 | 0.958 | 0.869 | 0.817 | 0.817 |
| APTOS | RETFound x Ordinal CapsNet + LoRA | 0.992 | 0.794 | 0.953 | 0.940 | 0.824 | 0.794 |
| Messidor-2 | RETFound x Ordinal CapsNet | 0.931 | 0.916 | 0.789 | 0.698 | 0.750 | 0.698 |
| Messidor-2 | RETFound x MLP + K-1 sigmoid | 0.923 | 0.931 | 0.839 | 0.767 | 0.750 | 0.750 |
| Messidor-2 | DINOv2  x Ordinal CapsNet | 0.962 | 0.870 | 0.728 | 1.000 | 0.583 | 0.583 |
| Messidor-2 | DINOv2  x MLP + K-1 sigmoid | 0.964 | 0.832 | 0.778 | 0.930 | 0.500 | 0.500 |

### CCP (Mondrian, per-class quantile)

| Dataset | Model | C0 | C1 | C2 | C3 | C4 | min | CCP \|S\| |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| APTOS | RETFound x Ordinal CapsNet | 0.913 | 0.910 | 0.915 | 0.915 | 0.901 | 0.901 | 1.84 |
| APTOS | RETFound x MLP + K-1 sigmoid | 0.907 | 0.908 | 0.907 | 0.926 | 0.898 | 0.898 | 1.84 |
| APTOS | DINOv2  x Ordinal CapsNet | 0.911 | 0.901 | 0.901 | 0.922 | 0.916 | 0.901 | 1.75 |
| APTOS | DINOv2  x MLP + K-1 sigmoid | 0.925 | 0.880 | 0.896 | 0.912 | 0.893 | 0.880 | 1.66 |
| APTOS | RETFound x Ordinal CapsNet + LoRA | 0.931 | 0.899 | 0.904 | 0.912 | 0.911 | 0.899 | 1.67 |
| Messidor-2 | RETFound x Ordinal CapsNet | 0.911 | 0.954 | 0.894 | 0.977 | 0.833 | 0.833 | 3.44 |
| Messidor-2 | RETFound x MLP + K-1 sigmoid | 0.913 | 0.916 | 0.900 | 0.930 | 0.833 | 0.833 | 3.41 |
| Messidor-2 | DINOv2  x Ordinal CapsNet | 0.909 | 0.939 | 0.833 | 0.930 | 1.000 | 0.833 | 2.73 |
| Messidor-2 | DINOv2  x MLP + K-1 sigmoid | 0.929 | 0.931 | 0.822 | 1.000 | 0.917 | 0.822 | 2.54 |
