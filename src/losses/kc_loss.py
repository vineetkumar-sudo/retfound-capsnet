"""Differentiable Quadratic Weighted Kappa loss (KC Loss) for Ordinal CapsNet.

Ma et al. 2023: KC Loss directly optimizes Cohen's quadratic weighted kappa
via a differentiable approximation. We combine it as a secondary term:

    total = ordinal_margin_loss + gamma * kappa_loss

The ordinal margin loss already provides the classification signal, so we
use only the kappa-ranking term (L_KP) without the auxiliary L_CE.

Conversion from K-1 binary head outputs to a K-class distribution uses the
chain rule over positive-capsule lengths:

    P(y=0)   = 1 - p_0
    P(y=k)   = p_0 * ... * p_{k-1} * (1 - p_k)   for 0 < k < K-1
    P(y=K-1) = p_0 * ... * p_{K-2}

where p_k = length of head k's positive capsule. Sums to 1 exactly.
"""

from __future__ import annotations

import torch
import torch.nn as nn


def heads_to_class_probs(
    head_lengths: torch.Tensor, num_classes: int = 5, eps: float = 1e-6
) -> torch.Tensor:
    """(B, num_heads, 2) -> (B, num_classes) via chain rule.

    p_k = head_lengths[:, k, 1] (positive capsule length).
    """
    p = head_lengths[:, :, 1].clamp(min=eps, max=1.0 - eps)  # (B, K-1)
    q = 1.0 - p
    cum_p = torch.cumprod(p, dim=1)  # (B, K-1)

    B = head_lengths.size(0)
    ones = torch.ones(B, 1, device=p.device, dtype=p.dtype)
    prev = torch.cat([ones, cum_p[:, :-1]], dim=1)  # cumulative product up to (not incl) k

    mid = prev * q  # (B, K-1), gives P(y=0)..P(y=K-2)
    last = cum_p[:, -1:]  # P(y=K-1)
    return torch.cat([mid, last], dim=1)  # (B, K)


def quadratic_kappa_loss(
    probs: torch.Tensor, labels: torch.Tensor,
    num_classes: int = 5, eps: float = 1e-8,
) -> torch.Tensor:
    """Differentiable 1 - QWK. Minimize to push kappa toward 1.

    Formulation: L = num / den where
      num = sum_{i,j} w_{i,j} * (soft confusion matrix)_{i,j}
      den = sum_{i,j} w_{i,j} * (expected matrix under independence)_{i,j}
      w_{i,j} = (i - j)^2 / (K-1)^2
    """
    B = probs.size(0)
    idx = torch.arange(num_classes, device=probs.device, dtype=probs.dtype)
    W = ((idx.unsqueeze(0) - idx.unsqueeze(1)) ** 2) / ((num_classes - 1) ** 2)

    y_oh = torch.zeros(B, num_classes, device=probs.device, dtype=probs.dtype)
    y_oh.scatter_(1, labels.long().unsqueeze(1), 1.0)

    O = probs.T @ y_oh  # (K, K)  soft confusion matrix
    hist_pred = probs.sum(dim=0)
    hist_true = y_oh.sum(dim=0)
    E = torch.outer(hist_pred, hist_true) / B

    num = (W * O).sum()
    den = (W * E).sum() + eps
    return num / den


class KCLoss(nn.Module):
    """Kappa-ranking loss on K-1 binary head outputs (Ma et al. 2023 style)."""

    def __init__(self, num_classes: int = 5):
        super().__init__()
        self.num_classes = num_classes

    def forward(self, head_lengths: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        probs = heads_to_class_probs(head_lengths, self.num_classes)
        return quadratic_kappa_loss(probs, y, self.num_classes)
