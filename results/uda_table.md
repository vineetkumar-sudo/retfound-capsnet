# Unsupervised domain adaptation: APTOS -> Messidor-2

Ordinal CapsNet on frozen RETFound features, trained on APTOS only.
Feature alignment maps the target into the source space; label
correction reweights the decoded posterior. Everything except
`oracle` is label-free on the target. 5 folds, mean over folds.

| Align | Correct | QWK | Acc | Macro-F1 | MAE | rDR sens | rDR spec |
|---|---|---|---|---|---|---|---|
| none | rank(oracle) | 0.2619 ± 0.0414 | 0.4491 | 0.2943 | 0.865 | 0.384 | 0.781 |
| none | rank(src prior) | 0.2266 ± 0.0354 | 0.4000 | 0.2508 | 1.055 | 0.531 | 0.638 |
| CORAL | rank(oracle) | 0.2229 ± 0.0105 | 0.4606 | 0.2999 | 0.872 | 0.385 | 0.782 |
| z-score | rank(oracle) | 0.2212 ± 0.0178 | 0.4537 | 0.2914 | 0.881 | 0.370 | 0.776 |
| z-score | oracle | 0.1951 ± 0.0113 | 0.4451 | 0.2556 | 0.986 | 0.368 | 0.772 |
| z-score | rank(src prior) | 0.1924 ± 0.0147 | 0.4046 | 0.2521 | 1.071 | 0.508 | 0.630 |
| CORAL | oracle | 0.1882 ± 0.0141 | 0.4569 | 0.2881 | 0.942 | 0.403 | 0.737 |
| CORAL | rank(src prior) | 0.1860 ± 0.0105 | 0.4083 | 0.2473 | 1.075 | 0.523 | 0.636 |
| CORAL | none | 0.1742 ± 0.0211 | 0.4463 | 0.2505 | 1.082 | 0.467 | 0.687 |
| z-score | none | 0.1420 ± 0.0165 | 0.3954 | 0.2122 | 1.299 | 0.502 | 0.633 |
| CORAL | EM/SLD | 0.1399 ± 0.0384 | 0.3882 | 0.1933 | 1.422 | 0.496 | 0.653 |
| CORAL | BBSE | 0.1189 ± 0.0510 | 0.3250 | 0.1702 | 1.254 | 0.442 | 0.675 |
| z-score | BBSE | 0.0651 ± 0.0271 | 0.2537 | 0.1429 | 1.534 | 0.540 | 0.536 |
| none | BBSE | 0.0494 ± 0.0236 | 0.5080 | 0.1509 | 0.958 | 0.179 | 0.858 |
| none | none | 0.0132 ± 0.0090 | 0.5835 | 0.1559 | 0.759 | 0.005 | 1.000 |
| z-score | EM/SLD | 0.0061 ± 0.0053 | 0.0393 | 0.0233 | 3.133 | 0.982 | 0.031 |
| none | oracle | 0.0015 ± 0.0013 | 0.5831 | 0.1474 | 0.762 | 0.001 | 1.000 |
| none | EM/SLD | 0.0005 ± 0.0010 | 0.5831 | 0.1473 | 0.762 | 0.000 | 1.000 |
