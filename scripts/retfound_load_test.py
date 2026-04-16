"""RETFound load test: verify model loads and produces [1, 1024] feature vectors."""

import torch
import timm

WEIGHTS_PATH = "data/weights/RETFound_mae_natureCFP.pth"

# 1. Inspect checkpoint structure
print("Loading checkpoint...")
checkpoint = torch.load(WEIGHTS_PATH, map_location="cpu", weights_only=False)

if isinstance(checkpoint, dict):
    print(f"Checkpoint keys: {list(checkpoint.keys())}")
    if "model" in checkpoint:
        state_dict = checkpoint["model"]
    else:
        state_dict = checkpoint
else:
    state_dict = checkpoint

print(f"Number of parameters in state_dict: {len(state_dict)}")
print(f"First 10 keys: {list(state_dict.keys())[:10]}")
print(f"Last 5 keys: {list(state_dict.keys())[-5:]}")

# 2. Create ViT-Large and load weights
model = timm.create_model("vit_large_patch16_224", pretrained=False)
msg = model.load_state_dict(state_dict, strict=False)
print(f"\nMissing keys: {msg.missing_keys[:10]}{'...' if len(msg.missing_keys) > 10 else ''}")
print(f"Unexpected keys: {msg.unexpected_keys[:10]}{'...' if len(msg.unexpected_keys) > 10 else ''}")
model.eval()

# 3. Test forward pass
dummy = torch.randn(1, 3, 224, 224)
with torch.no_grad():
    features = model.forward_features(dummy)  # should be [1, 197, 1024]
    cls_token = features[:, 0]                # [1, 1024]

print(f"\nforward_features output: {features.shape}")
print(f"CLS token (feature vector): {cls_token.shape}")

assert cls_token.shape == (1, 1024), f"Expected [1, 1024], got {cls_token.shape}"
print("\n✓ RETFound loaded successfully — 1024-dim feature vector confirmed.")
