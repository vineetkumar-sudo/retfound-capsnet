"""Vanilla CapsNet head (Sabour-style) operating on cached RETFound features.

Architecture:
    (B, 1024) -> PrimaryCaps -> (B, 32, 8)
              -> DigitCaps (3-iter routing) -> (B, 5, 16)
              -> [optional] Decoder: masked capsule -> (B, 1024)

The `forward` method returns a dict with the capsule vectors, their lengths,
the final routing coefficients, and an optional reconstruction — everything
downstream code (loss, UQ, eval) needs.
"""

from __future__ import annotations

from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F


def squash(s: torch.Tensor, dim: int = -1, eps: float = 1e-8) -> torch.Tensor:
    """Sabour squash: scales a vector so its length is in (0, 1).

    v = (||s||^2 / (1 + ||s||^2)) * (s / ||s||)
    Epsilon inside the norm prevents NaN on short/zero vectors.
    """
    norm2 = (s * s).sum(dim=dim, keepdim=True)
    norm = torch.sqrt(norm2 + eps)
    return (norm2 / (1.0 + norm2)) * (s / norm)


def squash_nonuniform(s: torch.Tensor, dim: int = -1, eps: float = 1e-8) -> torch.Tensor:
    """Gogulamudi et al. 2024 non-uniform squash.

    v = (||s|| / (1 + ||s||)) * (s / ||s||) = s / (1 + ||s||)
    Saturation is linear in ||s|| rather than quadratic, so short vectors keep
    more of their direction signal while long vectors remain bounded to (0, 1).
    """
    norm = torch.sqrt((s * s).sum(dim=dim, keepdim=True) + eps)
    return s / (1.0 + norm)


_SQUASH_VARIANTS: dict[str, Callable[..., torch.Tensor]] = {
    "sabour": squash,
    "nonuniform": squash_nonuniform,
}


def squash_fn(variant: str) -> Callable[..., torch.Tensor]:
    """Dispatch a squash variant by name. Raises on unknown names — no silent fallback."""
    if variant not in _SQUASH_VARIANTS:
        raise ValueError(
            f"Unknown squash variant {variant!r}. Expected one of {sorted(_SQUASH_VARIANTS)}."
        )
    return _SQUASH_VARIANTS[variant]


class PrimaryCaps(nn.Module):
    """Projects a flat feature vector into `num_caps` capsules of `caps_dim`."""

    def __init__(
        self,
        in_dim: int = 1024,
        num_caps: int = 32,
        caps_dim: int = 8,
        squash_variant: str = "sabour",
    ):
        super().__init__()
        self.num_caps = num_caps
        self.caps_dim = caps_dim
        self.squash_variant = squash_variant
        self._squash = squash_fn(squash_variant)
        self.linear = nn.Linear(in_dim, num_caps * caps_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, in_dim) -> (B, num_caps, caps_dim)
        u = self.linear(x).view(-1, self.num_caps, self.caps_dim)
        return self._squash(u, dim=-1)


class DigitCaps(nn.Module):
    """Dynamic routing between primary capsules and class ("digit") capsules."""

    def __init__(
        self,
        num_primary: int = 32,
        primary_dim: int = 8,
        num_classes: int = 5,
        caps_dim: int = 16,
        routing_iters: int = 3,
        squash_variant: str = "sabour",
    ):
        super().__init__()
        self.num_primary = num_primary
        self.num_classes = num_classes
        self.primary_dim = primary_dim
        self.caps_dim = caps_dim
        self.routing_iters = routing_iters
        self.squash_variant = squash_variant
        self._squash = squash_fn(squash_variant)

        # Transformation tensors W_ij: one per (primary, digit) pair.
        # Shape (1, num_primary, num_classes, caps_dim, primary_dim) for batch broadcast.
        self.W = nn.Parameter(
            0.01 * torch.randn(1, num_primary, num_classes, caps_dim, primary_dim)
        )

    def forward(
        self, u: torch.Tensor, return_routing_history: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            u: (B, num_primary, primary_dim)
            return_routing_history: if True, also return a stack of coupling
                coefficients from each routing iteration for UQ use.
        Returns:
            v: (B, num_classes, caps_dim) — final digit capsule vectors
            c: (B, num_primary, num_classes) — final routing coefficients
            [optional] c_history: (routing_iters, B, num_primary, num_classes)
        """
        B = u.size(0)
        # u: (B, num_primary, primary_dim) -> (B, num_primary, 1, primary_dim, 1)
        u = u.unsqueeze(2).unsqueeze(-1)
        # u_hat: (B, num_primary, num_classes, caps_dim, 1) -> (B, num_primary, num_classes, caps_dim)
        u_hat = torch.matmul(self.W, u).squeeze(-1)
        # Detach for all routing iters except the last — routing is an inference-time
        # procedure; gradients flow through the final iteration only (Sabour et al.).
        u_hat_detached = u_hat.detach()

        b = torch.zeros(B, self.num_primary, self.num_classes, device=u.device)

        v = torch.zeros(B, self.num_classes, self.caps_dim, device=u.device)
        c = torch.zeros_like(b)
        history: list[torch.Tensor] = [] if return_routing_history else []
        for r in range(self.routing_iters):
            c = F.softmax(b, dim=2)  # softmax over digit-caps dim
            if return_routing_history:
                history.append(c.detach())
            uh = u_hat if r == self.routing_iters - 1 else u_hat_detached
            # s_j = sum_i c_ij * u_hat_ij : (B, num_classes, caps_dim)
            s = (c.unsqueeze(-1) * uh).sum(dim=1)
            v = self._squash(s, dim=-1)
            if r < self.routing_iters - 1:
                # agreement a_ij = u_hat_ij . v_j  ->  b += a
                agreement = (uh * v.unsqueeze(1)).sum(dim=-1)
                b = b + agreement

        if return_routing_history:
            c_hist = torch.stack(history, dim=0)  # (R, B, num_primary, num_classes)
            return v, c, c_hist
        return v, c


class Decoder(nn.Module):
    """Optional reconstruction decoder: one 16-dim capsule -> 1024-dim feature.

    During training, the correct class's capsule is selected via the label.
    At eval, the longest capsule is used.
    """

    def __init__(
        self,
        caps_dim: int = 16,
        num_classes: int = 5,
        out_dim: int = 1024,
        hidden: int = 512,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(caps_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, out_dim),
        )

    def forward(
        self,
        digit_caps: torch.Tensor,
        labels: torch.Tensor | None,
        lengths: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            digit_caps: (B, num_classes, caps_dim)
            labels:     (B,) int64 or None (use lengths.argmax at eval time)
            lengths:    (B, num_classes) — capsule vector lengths
        Returns:
            (B, out_dim) reconstructed features
        """
        if labels is None:
            idx = lengths.argmax(dim=1)
        else:
            idx = labels
        B = digit_caps.size(0)
        selected = digit_caps[torch.arange(B, device=digit_caps.device), idx]
        return self.net(selected)


class CapsNet(nn.Module):
    """Vanilla 5-class CapsNet head for DR grading on 1024-dim RETFound features."""

    def __init__(
        self,
        feature_dim: int = 1024,
        num_primary: int = 32,
        primary_dim: int = 8,
        num_classes: int = 5,
        caps_dim: int = 16,
        routing_iters: int = 3,
        use_decoder: bool = False,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.use_decoder = use_decoder

        self.primary = PrimaryCaps(feature_dim, num_primary, primary_dim)
        self.digit = DigitCaps(
            num_primary, primary_dim, num_classes, caps_dim, routing_iters
        )
        self.decoder = (
            Decoder(caps_dim, num_classes, feature_dim) if use_decoder else None
        )

    def forward(
        self,
        x: torch.Tensor,
        labels: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        u = self.primary(x)
        digit_caps, coupling = self.digit(u)
        lengths = digit_caps.norm(dim=-1)  # (B, num_classes)

        out: dict[str, torch.Tensor] = {
            "digit_caps": digit_caps,
            "lengths": lengths,
            "coupling": coupling,
        }
        if self.decoder is not None:
            out["reconstruction"] = self.decoder(digit_caps, labels, lengths)
        return out
