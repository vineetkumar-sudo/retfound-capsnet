"""MLP + K-1 sigmoid ordinal head — the direct no-capsule comparator.

Ablation target: isolate the contribution of the capsule machinery (PrimaryCaps
+ routing-by-agreement DigitCaps) from the K-1 binary ordinal decomposition.
This head uses the same decomposition as OrdinalCapsNet but replaces every
capsule operation with a plain MLP + per-head sigmoid on the 1024-dim
RETFound CLS token.

Design:
    - Shared trunk:  Linear(feature_dim -> hidden)  -> ReLU  -> Dropout
                     -> Linear(hidden -> hidden)    -> ReLU  -> Dropout
    - K-1 heads:     independent Linear(hidden -> 1), each followed by sigmoid
    - Output:        the same `head_lengths` (B, K-1, 2) dict schema that
                     OrdinalCapsNet returns, so OrdinalMarginLoss /
                     predict_grade_from_heads / count_non_monotonic /
                     collect_preds plug in unchanged.

The per-head probability goes into `head_lengths[:, k, 1]` and its complement
into `head_lengths[:, k, 0]`, matching Sabour's two-capsule convention where
[..., 0] is the "y <= k" capsule and [..., 1] is the "y > k" capsule.

Parameter count with the defaults below (feature_dim=1024, hidden=256,
K-1=4): 1024*256 + 256 + 256*256 + 256 + 4*(256 + 1) = 330_500 trainable
parameters, which is deliberately close to OrdinalCapsNet's 295K so the
comparison is a fair head-vs-head ablation rather than a capacity test.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class MLPOrdinal(nn.Module):
    def __init__(
        self,
        feature_dim: int = 1024,
        hidden_dim: int = 256,
        num_classes: int = 5,
        num_hidden_layers: int = 2,
        dropout: float = 0.1,
        # The kwargs below are accepted but ignored. They exist so the
        # existing run_ordinal_capsnet.py::build_model can dispatch to this
        # class without special-casing its config payload (num_primary etc.).
        num_primary: int | None = None,
        primary_dim: int | None = None,
        caps_dim: int | None = None,
        routing_iters: int | None = None,
        squash_variant: str | None = None,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.num_heads = num_classes - 1
        self.hidden_dim = hidden_dim

        trunk: list[nn.Module] = []
        in_dim = feature_dim
        for _ in range(num_hidden_layers):
            trunk.append(nn.Linear(in_dim, hidden_dim))
            trunk.append(nn.ReLU(inplace=True))
            if dropout > 0:
                trunk.append(nn.Dropout(dropout))
            in_dim = hidden_dim
        self.trunk = nn.Sequential(*trunk)

        self.heads = nn.ModuleList(
            [nn.Linear(hidden_dim, 1) for _ in range(self.num_heads)]
        )

    def forward(
        self, x: torch.Tensor, return_routing_history: bool = False
    ) -> dict[str, torch.Tensor | list]:
        h = self.trunk(x)                                    # (B, hidden_dim)
        logits = torch.cat([head(h) for head in self.heads], dim=1)   # (B, K-1)
        p_gt = torch.sigmoid(logits)                         # (B, K-1)
        # Schema-compatible two-capsule-length tensor: [:, :, 0] = 1 - P(y>k),
        # [:, :, 1] = P(y>k). Downstream code reads [:, :, 1] as the positive
        # probability, so the loss + inference paths stay identical.
        head_lengths = torch.stack([1.0 - p_gt, p_gt], dim=-1)        # (B, K-1, 2)

        out: dict = {"head_lengths": head_lengths}
        if return_routing_history:
            # No routing here. Emit a zero tensor in the same (H, R, B, P, 2)
            # layout DigitCaps uses, so collect_preds' routing-variance reducer
            # returns exact zeros per sample. R must be >= 2 because the
            # downstream `rh.std(dim=1)` uses unbiased=True and would return
            # NaN on a size-1 dim; R=2 gives a clean zero.
            B = x.shape[0]
            out["routing_history"] = head_lengths.new_zeros(
                (self.num_heads, 2, B, 1, 2)
            )
        return out
