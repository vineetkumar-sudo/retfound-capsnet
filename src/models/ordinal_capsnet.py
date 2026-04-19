"""Ordinal CapsNet: K-1 independent binary heads with dynamic routing.

Architecture:
    (B, 1024) -> PrimaryCaps -> (B, 32, 8)
              -> for each k in 0..K-2: DigitCaps (2 caps x 16-dim, 3-iter routing)
                    -> head_k outputs P(y > k) via capsule lengths

All heads share the same PrimaryCaps but have independent DigitCaps weights.
The decomposition respects DR grading as an ordinal problem while keeping
capsule-based routing per binary decision.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from src.models.capsnet import DigitCaps, PrimaryCaps


class OrdinalCapsNet(nn.Module):
    def __init__(
        self,
        feature_dim: int = 1024,
        num_primary: int = 32,
        primary_dim: int = 8,
        num_classes: int = 5,
        caps_dim: int = 16,
        routing_iters: int = 3,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.num_heads = num_classes - 1
        self.caps_dim = caps_dim

        self.primary = PrimaryCaps(feature_dim, num_primary, primary_dim)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # K-1 independent binary heads — each a DigitCaps with 2 output capsules
        self.heads = nn.ModuleList([
            DigitCaps(
                num_primary=num_primary,
                primary_dim=primary_dim,
                num_classes=2,
                caps_dim=caps_dim,
                routing_iters=routing_iters,
            )
            for _ in range(self.num_heads)
        ])

    def forward(
        self, x: torch.Tensor, return_routing_history: bool = False
    ) -> dict[str, torch.Tensor | list]:
        u = self.primary(x)
        u = self.dropout(u)

        head_caps: list[torch.Tensor] = []
        head_couplings: list[torch.Tensor] = []
        head_lengths: list[torch.Tensor] = []
        head_histories: list[torch.Tensor] = []
        for head in self.heads:
            if return_routing_history:
                v, c, c_hist = head(u, return_routing_history=True)
                head_histories.append(c_hist)
            else:
                v, c = head(u)
            head_caps.append(v)
            head_couplings.append(c)
            head_lengths.append(v.norm(dim=-1))

        out: dict = {
            "head_caps": torch.stack(head_caps, dim=1),        # (B, num_heads, 2, caps_dim)
            "head_lengths": torch.stack(head_lengths, dim=1),  # (B, num_heads, 2)
            "head_couplings": head_couplings,                  # list of (B, num_primary, 2)
        }
        if return_routing_history:
            # (num_heads, routing_iters, B, num_primary, 2)
            out["routing_history"] = torch.stack(head_histories, dim=0)
        return out
