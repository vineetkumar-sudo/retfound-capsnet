"""RETFound (ViT-L) + LoRA + Ordinal CapsNet head, end-to-end trainable.

Architecture:
    Raw fundus image (224×224×3)
      → RETFound ViT-L (frozen weights + LoRA adapters on attn.qkv)
      → CLS token  (1024-dim)
      → PrimaryCaps (32 × 8, squash)
      → 4 binary ordinal DigitCaps heads (2 × 16-dim each)

Only the LoRA adapter matrices + the Ordinal CapsNet head carry gradients; the
original ViT weights remain frozen. Targets the fused `attn.qkv` Linear in every
ViT block (rank 8, alpha 16, dropout 0.05 → ~787K LoRA params + ~295K head
= ~1.08M trainable).
"""

from __future__ import annotations

from pathlib import Path

import timm
import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model

from src.models.ordinal_capsnet import OrdinalCapsNet


def load_retfound_backbone(weights_path: str) -> nn.Module:
    """Load the RETFound MAE checkpoint into a timm ViT-L backbone."""
    backbone = timm.create_model("vit_large_patch16_224", pretrained=False)
    ckpt = torch.load(weights_path, map_location="cpu", weights_only=False)
    state_dict = ckpt["model"] if "model" in ckpt else ckpt
    # strict=False is safe here — mismatches are the MAE decoder + ImageNet classification head;
    # 24 ViT-L encoder blocks all map cleanly (verified in src/data/feature_cache.py setup).
    backbone.load_state_dict(state_dict, strict=False)
    return backbone


class RetfoundLoraOrdinalCapsNet(nn.Module):
    def __init__(
        self,
        weights_path: str = "data/weights/RETFound_mae_natureCFP.pth",
        lora_r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.05,
        feature_dim: int = 1024,
        num_classes: int = 5,
        num_primary: int = 32,
        primary_dim: int = 8,
        caps_dim: int = 16,
        routing_iters: int = 3,
        head_dropout: float = 0.1,
    ):
        super().__init__()
        if not Path(weights_path).exists():
            raise FileNotFoundError(f"Missing RETFound weights at {weights_path}")

        backbone = load_retfound_backbone(weights_path)
        # Strip the ImageNet classifier head — we only need features.
        backbone.reset_classifier(0)

        lora_cfg = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=["qkv"],        # matches blocks.N.attn.qkv in every block
            lora_dropout=lora_dropout,
            bias="none",
        )
        self.backbone = get_peft_model(backbone, lora_cfg)

        self.head = OrdinalCapsNet(
            feature_dim=feature_dim, num_classes=num_classes,
            num_primary=num_primary, primary_dim=primary_dim,
            caps_dim=caps_dim, routing_iters=routing_iters,
            dropout=head_dropout,
        )

    def forward(self, x: torch.Tensor, return_routing_history: bool = False) -> dict:
        # forward_features returns (B, 1+N_patches, dim) in timm ViT
        feats = self.backbone.forward_features(x)   # (B, 197, 1024)
        cls = feats[:, 0]                            # (B, 1024)
        return self.head(cls, return_routing_history=return_routing_history)

    def trainable_params_report(self) -> dict:
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen = sum(p.numel() for p in self.parameters() if not p.requires_grad)
        return {"trainable": trainable, "frozen": frozen, "total": trainable + frozen}

    def set_tuning_mode(self, mode: str, unfreeze_blocks: int = 0) -> dict:
        """Switch between LoRA, full fine-tuning, and progressive unfreezing.

        Reviewer comment 6 asked why full fine-tuning and layer-wise
        progressive unfreezing were excluded from the main comparison. This
        makes both runnable from the same driver, so the three tiers share a
        protocol and differ only in which backbone parameters receive
        gradients.

            "lora"        adapters only (default; leaves the model as built)
            "full"        every backbone parameter trainable
            "progressive" only the last `unfreeze_blocks` transformer blocks
                          (plus the final norm), the classic top-down schedule

        The head is always trainable. Returns the parameter-count report.
        """
        assert mode in ("lora", "full", "progressive"), f"bad mode: {mode}"
        if mode == "lora":
            return self.trainable_params_report()

        # PEFT wraps the timm model; reach the real ViT underneath.
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
                raise AttributeError("could not locate transformer blocks on backbone")
            n = len(blocks)
            keep = blocks[max(0, n - unfreeze_blocks):] if unfreeze_blocks > 0 else []
            for blk in keep:
                for p in blk.parameters():
                    p.requires_grad_(True)
            for attr in ("norm", "fc_norm"):
                mod = getattr(vit, attr, None)
                if mod is not None:
                    for p in mod.parameters():
                        p.requires_grad_(True)

        for p in self.head.parameters():
            p.requires_grad_(True)
        return self.trainable_params_report()
