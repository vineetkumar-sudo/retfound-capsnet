"""Direction-aware asymmetric ordinal margin loss for CapsNet K-1 heads.

Extension of OrdinalMarginLoss. Rationale:

  Under-grading (predicting lower severity than truth) is clinically dangerous —
  the patient misses follow-up while disease progresses.
  Over-grading is merely inconvenient — an extra screening visit.

  So we weight per-sample error direction differently: samples whose truth says
  "y > k" (under-grading candidates) get multiplied by lambda_under; samples
  whose truth says "y <= k" (over-grading candidates) get lambda_over.

Setting lambda_under == lambda_over == 1.0 recovers the symmetric loss exactly.

Independent of direction weighting, m_plus and m_minus can be tightened
(e.g., 0.95/0.05) to demand more confident correct-capsule activation.

Per-head task-importance (Niu 2016) is supported via `per_head_weights`.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.losses.ordinal_loss import ordinal_labels


class AsymmetricOrdinalMarginLoss(nn.Module):
    def __init__(
        self,
        num_classes: int = 5,
        m_plus: float = 0.9,
        m_minus: float = 0.1,
        lambda_base: float = 0.5,
        lambda_under: float = 2.0,
        lambda_over: float = 1.0,
        per_head_weights: list[float] | None = None,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.num_heads = num_classes - 1
        self.m_plus = m_plus
        self.m_minus = m_minus
        self.lambda_base = lambda_base
        self.lambda_under = lambda_under
        self.lambda_over = lambda_over
        if per_head_weights is None:
            self.per_head_weights = [1.0] * self.num_heads
        else:
            assert len(per_head_weights) == self.num_heads, (
                f"per_head_weights must have length {self.num_heads}"
            )
            self.per_head_weights = per_head_weights

    def forward(
        self,
        head_lengths: torch.Tensor,
        y: torch.Tensor,
        return_per_head: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            head_lengths: (B, num_heads, 2) capsule lengths; [:, k, 0] = P(y<=k),
                          [:, k, 1] = P(y>k).
            y: (B,) int grade labels
            return_per_head: if True, also return (num_heads,) per-head losses
        """
        binary_labels = ordinal_labels(y, self.num_classes).long()  # (B, num_heads)
        total = head_lengths.new_zeros(())
        per_head: list[torch.Tensor] = []

        for k in range(self.num_heads):
            lengths_k = head_lengths[:, k, :]  # (B, 2)
            labels_k = binary_labels[:, k]     # (B,) in {0, 1}
            T = F.one_hot(labels_k, num_classes=2).float()  # (B, 2)

            # Standard symmetric margin terms (pre-direction-weighting)
            pos = T * torch.clamp(self.m_plus - lengths_k, min=0.0).pow(2)
            neg = self.lambda_base * (1.0 - T) * torch.clamp(
                lengths_k - self.m_minus, min=0.0
            ).pow(2)
            per_sample = (pos + neg).sum(dim=1)  # (B,)

            # Direction weight: under-grading samples (truth=1) * lambda_under,
            # over-grading samples (truth=0) * lambda_over
            direction = (
                labels_k.float() * self.lambda_under
                + (1.0 - labels_k.float()) * self.lambda_over
            )  # (B,)

            loss_k = (direction * per_sample).mean()
            per_head.append(loss_k)
            total = total + self.per_head_weights[k] * loss_k

        total = total / self.num_heads

        if return_per_head:
            return total, torch.stack(per_head)
        return total


def compute_niu_weights(labels: np.ndarray, num_classes: int) -> list[float]:
    """Niu 2016-style per-head task-importance: lambda_t proportional to sqrt(N_t).

    N_t = count of samples where y > t (positive count for head t).
    Weights are normalized so their MEAN = 1.0, preserving the absolute scale of
    the overall loss (only the relative weighting across heads changes).
    """
    num_heads = num_classes - 1
    raw = np.array(
        [np.sqrt((labels > t).sum()) for t in range(num_heads)], dtype=np.float64
    )
    # Normalize so mean = 1 (sum = num_heads)
    weights = raw / raw.mean()
    return weights.tolist()
