# Rank-monotonicity: constrained projection vs. clip-and-renormalise

A = product decode + clip + renormalise (as submitted).
B = cumulative-difference decode + clip + renormalise.
C = monotone projection (PAV) + cumulative-difference decode (proposed);
needs no clipping and no renormalisation -- rows sum to exactly 1.

`viol%` is the fraction of samples whose CONTINUOUS heads violate
monotonicity, i.e. the fraction the projection actually moves. This is
distinct from the paper's `count_non_monotonic`, which only counts
contradictions among the thresholded h > 0.5 decisions.

| Dataset | Model | viol% | ECE A | ECE B | ECE C | QWK A | QWK B | QWK C | Acc A | Acc C | Renorm% A | Renorm% C |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| APTOS | RETFound x Ordinal CapsNet | 55.6% | 0.1469 | 0.0826 | **0.0715** | 0.8806 | 0.8720 | **0.8721** | 0.8015 | **0.7975** | 100.0% | 0.0% |
| APTOS | RETFound x MLP + K-1 sigmoid | 46.2% | 0.1026 | 0.0565 | **0.0552** | 0.8766 | 0.8729 | **0.8730** | 0.8051 | **0.8057** | 86.0% | 0.0% |
| APTOS | DINOv2  x Ordinal CapsNet | 56.7% | 0.1238 | 0.0700 | **0.0580** | 0.8966 | 0.8886 | **0.8881** | 0.8097 | **0.8082** | 100.0% | 0.0% |
| APTOS | DINOv2  x MLP + K-1 sigmoid | 44.5% | 0.1060 | 0.0584 | **0.0562** | 0.8944 | 0.8862 | **0.8862** | 0.8147 | **0.8151** | 100.0% | 0.0% |
| APTOS | RETFound x Ordinal CapsNet + LoRA | 56.2% | 0.1339 | 0.0669 | **0.0549** | 0.9071 | 0.9037 | **0.9036** | 0.8283 | **0.8262** | 100.0% | 0.0% |
| Messidor-2 | RETFound x Ordinal CapsNet | 37.8% | 0.1054 | 0.0494 | **0.0459** | 0.5520 | 0.5301 | **0.5377** | 0.6044 | **0.6181** | 100.0% | 0.0% |
| Messidor-2 | RETFound x MLP + K-1 sigmoid | 4.2% | 0.0860 | 0.0377 | **0.0387** | 0.5490 | 0.5345 | **0.5345** | 0.6101 | **0.6216** | 100.0% | 0.0% |
| Messidor-2 | DINOv2  x Ordinal CapsNet | 62.2% | 0.1207 | 0.0694 | **0.0580** | 0.7301 | 0.7235 | **0.7246** | 0.6938 | **0.6932** | 100.0% | 0.0% |
| Messidor-2 | DINOv2  x MLP + K-1 sigmoid | 48.5% | 0.0835 | 0.0307 | **0.0329** | 0.7515 | 0.7438 | **0.7437** | 0.7041 | **0.7087** | 99.9% | 0.0% |
