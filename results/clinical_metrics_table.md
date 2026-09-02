# Clinical screening metrics: referable and vision-threatening DR

Operating points on the ICDR scale, scored from the native K-1 head
output P(y > k) -- no extra thresholding is required, since head k is
already the binary detector for that level.

NHS diabetic eye screening bar for referable DR: sensitivity >= 0.85,
specificity >= 0.80. Abramoff et al. (IDx-DR) report 0.872 / 0.907.
95% CIs are bootstrap percentile intervals (500 resamples).

## Referable DR (grade >= 2) — the deployment-relevant threshold

| Dataset | Model | Prev. | AUROC (95% CI) | Sens (95% CI) | Spec (95% CI) | PPV | NPV | Bal. acc | NHS |
|---|---|---|---|---|---|---|---|---|---|
| APTOS | RETFound x Ordinal CapsNet | 0.406 | 0.9763 (0.974-0.979) | 0.948 (0.940-0.955) | 0.909 (0.901-0.917) | 0.877 | 0.962 | 0.928 | PASS |
| APTOS | RETFound x MLP + K-1 sigmoid | 0.406 | 0.9773 (0.975-0.980) | 0.939 (0.932-0.946) | 0.913 (0.906-0.921) | 0.881 | 0.956 | 0.926 | PASS |
| APTOS | DINOv2  x Ordinal CapsNet | 0.406 | 0.9784 (0.976-0.981) | 0.957 (0.951-0.963) | 0.914 (0.906-0.921) | 0.884 | 0.969 | 0.936 | PASS |
| APTOS | DINOv2  x MLP + K-1 sigmoid | 0.406 | 0.9808 (0.978-0.983) | 0.954 (0.948-0.960) | 0.916 (0.909-0.923) | 0.885 | 0.967 | 0.935 | PASS |
| APTOS | RETFound x Ordinal CapsNet + LoRA | 0.406 | 0.9808 (0.978-0.983) | 0.954 (0.947-0.961) | 0.931 (0.923-0.937) | 0.904 | 0.967 | 0.942 | PASS |
| Messidor-2 | RETFound x Ordinal CapsNet | 0.262 | 0.8196 (0.796-0.841) | 0.639 (0.596-0.679) | 0.853 (0.831-0.871) | 0.607 | 0.869 | 0.746 | FAIL |
| Messidor-2 | RETFound x MLP + K-1 sigmoid | 0.262 | 0.8250 (0.803-0.844) | 0.619 (0.578-0.661) | 0.852 (0.829-0.870) | 0.597 | 0.863 | 0.735 | FAIL |
| Messidor-2 | DINOv2  x Ordinal CapsNet | 0.262 | 0.8926 (0.871-0.912) | 0.720 (0.675-0.761) | 0.949 (0.936-0.961) | 0.835 | 0.905 | 0.835 | FAIL |
| Messidor-2 | DINOv2  x MLP + K-1 sigmoid | 0.262 | 0.9150 (0.897-0.931) | 0.744 (0.701-0.784) | 0.928 (0.912-0.942) | 0.785 | 0.911 | 0.836 | FAIL |

## All operating points (AUROC / sens / spec at head >= 0.5)

| Dataset | Model | any DR | rDR (>=2) | vtDR (>=3) | PDR (>=4) |
|---|---|---|---|---|---|
| APTOS | RETFound x Ordinal CapsNet | 0.991 / 0.97 / 0.97 | 0.976 / 0.95 / 0.91 | 0.931 / 0.70 / 0.95 | 0.897 / 0.50 / 0.97 |
| APTOS | RETFound x MLP + K-1 sigmoid | 0.992 / 0.97 / 0.97 | 0.977 / 0.94 / 0.91 | 0.929 / 0.64 / 0.95 | 0.897 / 0.43 / 0.98 |
| APTOS | DINOv2  x Ordinal CapsNet | 0.997 / 0.99 / 0.97 | 0.978 / 0.96 / 0.91 | 0.941 / 0.72 / 0.94 | 0.911 / 0.53 / 0.98 |
| APTOS | DINOv2  x MLP + K-1 sigmoid | 0.997 / 0.99 / 0.98 | 0.981 / 0.95 / 0.92 | 0.942 / 0.68 / 0.95 | 0.921 / 0.46 / 0.98 |
| APTOS | RETFound x Ordinal CapsNet + LoRA | 0.994 / 0.98 / 0.98 | 0.981 / 0.95 / 0.93 | 0.943 / 0.73 / 0.95 | 0.923 / 0.59 / 0.98 |
| Messidor-2 | RETFound x Ordinal CapsNet | 0.787 / 0.76 / 0.70 | 0.820 / 0.64 / 0.85 | 0.909 / 0.46 / 0.98 | 0.914 / 0.31 / 1.00 |
| Messidor-2 | RETFound x MLP + K-1 sigmoid | 0.788 / 0.72 / 0.73 | 0.825 / 0.62 / 0.85 | 0.909 / 0.47 / 0.98 | 0.905 / 0.31 / 1.00 |
| Messidor-2 | DINOv2  x Ordinal CapsNet | 0.842 / 0.73 / 0.83 | 0.893 / 0.72 / 0.95 | 0.981 / 0.82 / 0.98 | 0.908 / 0.43 / 1.00 |
| Messidor-2 | DINOv2  x MLP + K-1 sigmoid | 0.846 / 0.70 / 0.86 | 0.915 / 0.74 / 0.93 | 0.977 / 0.85 / 0.97 | 0.903 / 0.46 / 0.99 |

## Referable DR at a fixed clinical target

Left: highest specificity achievable at sensitivity >= 0.85. Right: highest sensitivity at specificity >= 0.80.

| Dataset | Model | Spec @ sens>=0.85 | Thresh | Sens @ spec>=0.80 | Thresh |
|---|---|---|---|---|---|
| APTOS | RETFound x Ordinal CapsNet | 0.955 | 0.684 | 0.984 | 0.222 |
| APTOS | RETFound x MLP + K-1 sigmoid | 0.957 | 0.697 | 0.988 | 0.200 |
| APTOS | DINOv2  x Ordinal CapsNet | 0.962 | 0.757 | 0.991 | 0.127 |
| APTOS | DINOv2  x MLP + K-1 sigmoid | 0.959 | 0.738 | 0.996 | 0.115 |
| APTOS | RETFound x Ordinal CapsNet + LoRA | 0.966 | 0.801 | 0.990 | 0.110 |
| Messidor-2 | RETFound x Ordinal CapsNet | 0.611 | 0.279 | 0.694 | 0.437 |
| Messidor-2 | RETFound x MLP + K-1 sigmoid | 0.614 | 0.270 | 0.678 | 0.428 |
| Messidor-2 | DINOv2  x Ordinal CapsNet | 0.743 | 0.196 | 0.827 | 0.234 |
| Messidor-2 | DINOv2  x MLP + K-1 sigmoid | 0.817 | 0.259 | 0.856 | 0.248 |
