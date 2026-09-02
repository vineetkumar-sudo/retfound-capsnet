# Ordinal contiguous conformal prediction (OCP)

Five random 50:50 cal/test splits per cell (seeds 42/123/456/789/1011);
mean +/- std across splits. `non-contig%` is the fraction of prediction
sets that skip an intermediate grade (e.g. {0, 3}), which is incoherent
under an ordinal scale. OCP is 0% by construction.

## alpha = 0.10 (target coverage 90%)

| Dataset | Model | LAC non-contig% | APS non-contig% | RAPS non-contig% | APS cov | APS \|S\| | OCP cov | OCP \|S\| | OCP worst-class |
|---|---|---|---|---|---|---|---|---|---|
| APTOS | RETFound x Ordinal CapsNet | 5.09% | 12.18% | 12.62% | 0.9433 | 1.814 | 0.9426 | 1.880 | 0.7685 |
| APTOS | RETFound x MLP + K-1 sigmoid | 2.33% | 8.40% | 8.70% | 0.9391 | 1.866 | 0.9376 | 1.909 | 0.7343 |
| APTOS | DINOv2  x Ordinal CapsNet | 2.29% | 8.12% | 8.32% | 0.9483 | 1.761 | 0.9480 | 1.811 | 0.8042 |
| APTOS | DINOv2  x MLP + K-1 sigmoid | 1.78% | 7.29% | 7.66% | 0.9454 | 1.804 | 0.9464 | 1.840 | 0.7911 |
| APTOS | RETFound x Ordinal CapsNet + LoRA | 0.74% | 4.88% | 4.96% | 0.9423 | 1.692 | 0.9430 | 1.712 | 0.7943 |
| Messidor-2 | RETFound x Ordinal CapsNet | 5.87% | 12.82% | 12.94% | 0.9034 | 2.496 | 0.9009 | 2.540 | 0.6821 |
| Messidor-2 | RETFound x MLP + K-1 sigmoid | 0.92% | 3.81% | 3.83% | 0.9021 | 2.484 | 0.9064 | 2.510 | 0.6865 |
| Messidor-2 | DINOv2  x Ordinal CapsNet | 4.11% | 9.91% | 10.16% | 0.8950 | 2.031 | 0.8917 | 2.033 | 0.6310 |
| Messidor-2 | DINOv2  x MLP + K-1 sigmoid | 0.64% | 3.74% | 3.58% | 0.9044 | 2.031 | 0.9025 | 2.016 | 0.6033 |

## alpha = 0.05 (target coverage 95%)

| Dataset | Model | APS non-contig% | RAPS non-contig% | APS \|S\| | OCP cov | OCP \|S\| |
|---|---|---|---|---|---|---|
| APTOS | RETFound x Ordinal CapsNet | 15.48% | 16.25% | 2.221 | 0.9676 | 2.287 |
| APTOS | RETFound x MLP + K-1 sigmoid | 6.92% | 7.32% | 2.366 | 0.9665 | 2.394 |
| APTOS | DINOv2  x Ordinal CapsNet | 12.43% | 13.73% | 2.078 | 0.9727 | 2.190 |
| APTOS | DINOv2  x MLP + K-1 sigmoid | 9.68% | 10.73% | 2.199 | 0.9701 | 2.232 |
| APTOS | RETFound x Ordinal CapsNet + LoRA | 10.03% | 11.25% | 1.994 | 0.9657 | 2.054 |
| Messidor-2 | RETFound x Ordinal CapsNet | 14.22% | 14.36% | 2.974 | 0.9507 | 3.001 |
| Messidor-2 | RETFound x MLP + K-1 sigmoid | 3.26% | 3.42% | 2.995 | 0.9493 | 3.032 |
| Messidor-2 | DINOv2  x Ordinal CapsNet | 14.75% | 14.77% | 2.608 | 0.9468 | 2.570 |
| Messidor-2 | DINOv2  x MLP + K-1 sigmoid | 5.85% | 4.22% | 2.557 | 0.9537 | 2.567 |
