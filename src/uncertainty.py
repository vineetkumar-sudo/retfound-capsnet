"""Capsule-native uncertainty quantification for Ordinal CapsNet.

Three independent UQ signals, all derivable from a single forward pass with
return_routing_history=True:

  1. digit_cap_entropy — entropy over normalized positive-capsule lengths
     (distributional uncertainty in which head is "active")
  2. routing_agreement_variance — std of coupling coefficients across routing
     iterations (non-convergence = model uncertain about part-whole assignment)
  3. prediction_margin — 1 - (p_top1 - p_top2) from 5-class probabilities
     (tight margin = near-tie between two grades = uncertain)

Higher value = more uncertain in all three. Outputs are (B,) scalar tensors
on CPU so callers can stack and compare directly.
"""

from __future__ import annotations

import torch

from src.losses.kc_loss import heads_to_class_probs


@torch.no_grad()
def digit_cap_entropy(head_lengths: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """(B, num_heads, 2) -> (B,) Shannon entropy of normalized positive lengths."""
    p = head_lengths[:, :, 1]  # (B, num_heads)
    p_norm = p / (p.sum(dim=1, keepdim=True) + eps)
    entropy = -(p_norm * torch.log(p_norm + eps)).sum(dim=1)
    return entropy.cpu()


@torch.no_grad()
def routing_agreement_variance(routing_history: torch.Tensor) -> torch.Tensor:
    """
    Args:
        routing_history: (num_heads, routing_iters, B, num_primary, 2) coupling
                         coefficients across routing iterations.
    Returns:
        (B,) mean std across heads/primaries/capsules of the coupling over iters.
    """
    # Shape: (H, R, B, P, 2). Std over R axis -> (H, B, P, 2), then mean across H, P, 2
    std_over_iters = routing_history.std(dim=1)  # (H, B, P, 2)
    # Average across heads, primaries, output capsules -> (B,)
    scalar = std_over_iters.mean(dim=(0, 2, 3))
    return scalar.cpu()


@torch.no_grad()
def prediction_margin(head_lengths: torch.Tensor, num_classes: int = 5) -> torch.Tensor:
    """1 - (top1 - top2) over 5-class probabilities (via chain rule).
    Higher = tighter margin = more uncertain."""
    probs = heads_to_class_probs(head_lengths, num_classes=num_classes)
    top2 = torch.topk(probs, k=2, dim=1).values  # (B, 2)
    margin = (top2[:, 0] - top2[:, 1]).clamp(min=0.0, max=1.0)
    return (1.0 - margin).cpu()


@torch.no_grad()
def compute_all_uq(model, X: torch.Tensor, device: torch.device) -> dict:
    """Run one forward pass with routing history and compute all 3 UQ methods."""
    model.eval()
    X = X.to(device)
    out = model(X, return_routing_history=True)
    return {
        "digit_entropy": digit_cap_entropy(out["head_lengths"]),
        "routing_variance": routing_agreement_variance(out["routing_history"]),
        "prediction_margin": prediction_margin(out["head_lengths"]),
        "head_lengths": out["head_lengths"].cpu(),
    }
