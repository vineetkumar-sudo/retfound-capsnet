"""DINOv2 ViT-L/14 + LoRA adapters + ordinal head (reviewer R1-12).

Reviewer comment 12 noted the comparison matrix was incomplete because no
DINOv2 LoRA cell was run ("essential to complete the comparison matrix").
This is the DINOv2 counterpart of `retfound_lora_capsnet.py`, kept
deliberately parallel so the only difference between the two LoRA rows is the
backbone.

DINOv2 is loaded from torch.hub rather than timm, and its hub wrapper returns
the CLS token directly from `forward()`, so no `forward_features` slicing is
needed. Its blocks expose `attn.qkv` under the same name as timm's ViT, so
the LoRA target spec is identical.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model

from src.models.mlp_ordinal import MLPOrdinal
from src.models.ordinal_capsnet import OrdinalCapsNet


class Dinov2LoraOrdinalCapsNet(nn.Module):
    def __init__(
        self,
        lora_r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.05,
        feature_dim: int = 1024,
        num_classes: int = 5,
        head_arch: str = "capsnet",
        num_primary: int = 32,
        primary_dim: int = 8,
        caps_dim: int = 16,
        routing_iters: int = 3,
        head_dropout: float = 0.1,
    ):
        super().__init__()
        backbone = torch.hub.load("facebookresearch/dinov2", "dinov2_vitl14")

        lora_cfg = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=["qkv"],      # blocks.N.attn.qkv, same as timm ViT
            lora_dropout=lora_dropout,
            bias="none",
        )
        self.backbone = get_peft_model(backbone, lora_cfg)

        if head_arch == "capsnet":
            self.head = OrdinalCapsNet(
                feature_dim=feature_dim, num_classes=num_classes,
                num_primary=num_primary, primary_dim=primary_dim,
                caps_dim=caps_dim, routing_iters=routing_iters,
                dropout=head_dropout,
            )
        elif head_arch == "mlp":
            self.head = MLPOrdinal(feature_dim=feature_dim,
                                   num_classes=num_classes, dropout=head_dropout)
        else:
            raise ValueError(f"unknown head_arch: {head_arch}")

    def forward(self, x: torch.Tensor, return_routing_history: bool = False) -> dict:
        # The DINOv2 hub wrapper's forward() already returns the CLS token.
        cls = self.backbone(x)                       # (B, 1024)
        if isinstance(self.head, OrdinalCapsNet):
            return self.head(cls, return_routing_history=return_routing_history)
        return self.head(cls)

    def trainable_params_report(self) -> dict:
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen = sum(p.numel() for p in self.parameters() if not p.requires_grad)
        return {"trainable": trainable, "frozen": frozen, "total": trainable + frozen}

    def set_tuning_mode(self, mode: str, unfreeze_blocks: int = 0) -> dict:
        """Mirror of the RETFound wrapper; see that docstring."""
        assert mode in ("lora", "full", "progressive"), f"bad mode: {mode}"
        if mode == "lora":
            return self.trainable_params_report()
        vit = getattr(self.backbone, "base_model", self.backbone)
        vit = getattr(vit, "model", vit)
        if mode == "full":
            for p in self.backbone.parameters():
                p.requires_grad_(True)
        else:
            for p in self.backbone.parameters():
                p.requires_grad_(False)
            blocks = getattr(vit, "blocks", None)
            if blocks is None:
                raise AttributeError("could not locate transformer blocks")
            n = len(blocks)
            for blk in blocks[max(0, n - unfreeze_blocks):]:
                for p in blk.parameters():
                    p.requires_grad_(True)
            norm = getattr(vit, "norm", None)
            if norm is not None:
                for p in norm.parameters():
                    p.requires_grad_(True)
        for p in self.head.parameters():
            p.requires_grad_(True)
        return self.trainable_params_report()
