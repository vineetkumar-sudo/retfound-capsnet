# Fine-tuning tiers on APTOS (reviewer R1-6)

Seed 42, 5-fold CV, identical protocol/splits; the tiers differ only in
which backbone parameters receive gradients. Mean +/- std across folds.
Wall-clock is the full 5-fold run on one A100.

| Tier | Trainable | QWK | Acc | Macro-F1 | MAE | 5-fold time |
|---|---:|---|---:|---:|---:|---:|
| Frozen head only | 295,168 | 0.8926 ± 0.0079 | 0.7900 | 0.6216 | nan | 2 min |
| LoRA r=8 | 1,081,600 | 0.9149 ± 0.0109 | 0.8270 | 0.6785 | 0.208 | 26 min |
| Progressive (last 4 blocks) | 50,813,184 | 0.9189 ± 0.0086 | 0.8310 | 0.6904 | 0.200 | 26 min |
| Full fine-tune | 304,383,232 | 0.8880 ± 0.0057 | 0.8033 | 0.6328 | 0.248 | 54 min |

Full fine-tuning is *worse* than training no backbone parameters at all
(0.8880 vs 0.8926) despite 1,031x more trainable weights: 2,636 training
images cannot support 304M parameters. Progressive unfreezing and LoRA are
statistically indistinguishable (0.9189 vs 0.9149, both fold-std ~0.01),
so LoRA reaches the same quality with 47x fewer trainable parameters.
The parameter-efficiency curve is non-monotonic, peaking well below full
capacity.
