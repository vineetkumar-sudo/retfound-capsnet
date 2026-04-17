"""K-1 binary ordinal decomposition for CapsNet heads.

For K classes {0, 1, ..., K-1}, encode each ordinal label y as K-1 binary targets:
    y_bin[k] = 1 if y > k else 0    (for k in 0..K-2)

So for K=5:
    y=0 -> [0,0,0,0]  (No DR)
    y=1 -> [1,0,0,0]  (Mild)
    y=2 -> [1,1,0,0]  (Moderate)
    y=3 -> [1,1,1,0]  (Severe)
    y=4 -> [1,1,1,1]  (PDR)

At inference, predicted grade = sum of heads where P(y > k) > threshold.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def ordinal_labels(y: torch.Tensor, num_classes: int) -> torch.Tensor:
    """Convert grade labels to K-1 binary "y > k" targets.

    Args:
        y: (B,) int tensor of grade labels in [0, num_classes-1]
        num_classes: total number of ordinal classes K
    Returns:
        (B, K-1) float tensor of 0/1.
    """
    ks = torch.arange(num_classes - 1, device=y.device)
    return (y.unsqueeze(-1) > ks).float()


class OrdinalMarginLoss(nn.Module):
    """Sum of K-1 per-head binary margin losses (Sabour margin applied per head).

    Each head's two capsules predict {y<=k, y>k}. We apply standard margin loss
    treating it as a 2-class problem, then average across heads.
    """

    def __init__(
        self,
        num_classes: int = 5,
        m_plus: float = 0.9,
        m_minus: float = 0.1,
        lambda_: float = 0.5,
        per_head_weights: list[float] | None = None,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.num_heads = num_classes - 1
        self.m_plus = m_plus
        self.m_minus = m_minus
        self.lambda_ = lambda_
        if per_head_weights is None:
            self.per_head_weights = [1.0] * self.num_heads
        else:
            assert len(per_head_weights) == self.num_heads
            self.per_head_weights = per_head_weights

    def forward(
        self,
        head_lengths: torch.Tensor,
        y: torch.Tensor,
        return_per_head: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            head_lengths: (B, num_heads, 2) — capsule vector lengths per head
                          head_lengths[:, k, 0] = P(y<=k) capsule length
                          head_lengths[:, k, 1] = P(y>k)  capsule length
            y: (B,) int grade labels
            return_per_head: if True, also return (num_heads,) per-head losses
        """
        binary_labels = ordinal_labels(y, self.num_classes).long()  # (B, num_heads)
        total = head_lengths.new_zeros(())
        per_head: list[torch.Tensor] = []
        for k in range(self.num_heads):
            lengths_k = head_lengths[:, k, :]  # (B, 2)
            T = F.one_hot(binary_labels[:, k], num_classes=2).float()  # (B, 2)
            pos = T * torch.clamp(self.m_plus - lengths_k, min=0.0).pow(2)
            neg = self.lambda_ * (1.0 - T) * torch.clamp(lengths_k - self.m_minus, min=0.0).pow(2)
            loss_k = (pos + neg).sum(dim=1).mean()
            per_head.append(loss_k)
            total = total + self.per_head_weights[k] * loss_k
        total = total / self.num_heads

        if return_per_head:
            return total, torch.stack(per_head)
        return total


def predict_grade_from_heads(
    head_lengths: torch.Tensor, threshold: float = 0.5
) -> torch.Tensor:
    """Aggregate K-1 binary head outputs into a single ordinal grade.

    Args:
        head_lengths: (B, num_heads, 2) capsule lengths
        threshold: decision threshold on the positive capsule's length
    Returns:
        (B,) predicted grades in {0, ..., num_classes-1}
    """
    p_gt_k = head_lengths[:, :, 1]  # (B, num_heads), positive-capsule length
    positives = (p_gt_k > threshold).long()
    return positives.sum(dim=1)


def count_non_monotonic(
    head_lengths: torch.Tensor, threshold: float = 0.5
) -> torch.Tensor:
    """Count samples whose binary decisions violate ordering.

    Valid monotonic patterns: [0,0,0,0], [1,0,0,0], [1,1,0,0], [1,1,1,0], [1,1,1,1].
    Invalid (e.g., [1,0,1,0]) means head 2 says "above Moderate" but head 1 says
    "below Mild" — contradictory since y>2 implies y>1.
    """
    p_gt_k = head_lengths[:, :, 1]
    positives = (p_gt_k > threshold).long()  # (B, K-1)
    diffs = positives[:, :-1] - positives[:, 1:]  # should be >= 0 for valid monotone
    violations = (diffs < 0).any(dim=1)
    return violations.sum()
