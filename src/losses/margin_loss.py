"""Sabour-style margin loss for CapsNet classification.

L_k = T_k * max(0, m+ - ||v_k||)^2  +  lambda * (1 - T_k) * max(0, ||v_k|| - m-)^2

Summed over classes, averaged over batch.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class MarginLoss(nn.Module):
    def __init__(
        self,
        m_plus: float = 0.9,
        m_minus: float = 0.1,
        lambda_: float = 0.5,
        num_classes: int = 5,
    ):
        super().__init__()
        self.m_plus = m_plus
        self.m_minus = m_minus
        self.lambda_ = lambda_
        self.num_classes = num_classes

    def forward(self, lengths: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            lengths: (B, num_classes) — capsule vector lengths in [0, 1)
            labels:  (B,) int64
        """
        T = F.one_hot(labels, num_classes=self.num_classes).float()  # (B, K)
        pos = T * torch.clamp(self.m_plus - lengths, min=0.0).pow(2)
        neg = self.lambda_ * (1.0 - T) * torch.clamp(lengths - self.m_minus, min=0.0).pow(2)
        return (pos + neg).sum(dim=1).mean()
