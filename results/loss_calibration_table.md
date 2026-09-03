# Loss x head factorial: what actually drives the miscalibration?

APTOS, frozen RETFound features, 3 seeds x 5-fold CV per cell.
Only the head architecture and the training loss vary; splits,
optimiser and decomposition are identical throughout.

`ECE(product)` uses the submitted decode (product + clip + renormalise);
`ECE(projected)` uses the monotone PAV projection + cumulative decode.
`viol%` is the rate of rank-monotonicity violations in the raw heads.

| Head | Loss | QWK | Acc | ECE (product) | ECE (projected) | viol% |
|---|---|---|---:|---:|---:|---:|
| Ordinal CapsNet | margin | 0.8922 ± 0.0013 | 0.7904 | 0.1581 ± 0.0069 | 0.0811 | 58.3% |
| Ordinal CapsNet | BCE | 0.8867 ± 0.0006 | 0.7853 | 0.0791 ± 0.0109 | 0.0323 | 50.3% |
| Ordinal CapsNet | focal | 0.8908 ± 0.0011 | 0.7943 | 0.2091 ± 0.0057 | 0.1122 | 46.9% |
| MLP + K-1 sigmoid | margin | 0.8847 ± 0.0008 | 0.7929 | 0.1050 ± 0.0051 | 0.0555 | 50.9% |
| MLP + K-1 sigmoid | BCE | 0.8856 ± 0.0016 | 0.7897 | 0.0181 ± 0.0018 | 0.0221 | 28.4% |
| MLP + K-1 sigmoid | focal | 0.8846 ± 0.0011 | 0.7896 | 0.1771 ± 0.0112 | 0.0904 | 19.3% |
