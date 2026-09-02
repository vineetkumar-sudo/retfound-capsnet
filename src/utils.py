"""Shared runtime helpers.

Currently just TF32 setup. Kept separate from the per-script `get_device()`
helpers so the policy lives in one place.
"""

from __future__ import annotations

import os

import torch

# Set once per process; re-entrant so every get_device() call is cheap.
_TF32_APPLIED = False


def tf32_enabled() -> bool:
    """TF32 is on by default; set RETFOUND_TF32=0 to opt out.

    The opt-out exists because TF32 rounds fp32 matmul inputs to 10 mantissa
    bits, so runs are NOT bit-identical to fp32 (or to the historical MPS
    runs). Use RETFOUND_TF32=0 when reproducing a logged number exactly.
    """
    return os.environ.get("RETFOUND_TF32", "1").strip().lower() not in {
        "0", "false", "no", "off",
    }


def enable_tf32(verbose: bool = True) -> bool:
    """Enable TF32 matmul/cuDNN paths on Ampere+ (SM >= 8.0). Returns whether on.

    Worth ~12x on fp32 matmul on an A100 vs the PyTorch default of full fp32.
    No-op on non-CUDA devices, on pre-Ampere cards, and when opted out.
    """
    global _TF32_APPLIED
    if _TF32_APPLIED or not torch.cuda.is_available():
        return _TF32_APPLIED

    if not tf32_enabled():
        if verbose:
            print("TF32: disabled via RETFOUND_TF32 (full fp32 precision)")
        return False

    if torch.cuda.get_device_capability()[0] < 8:
        if verbose:
            print("TF32: unavailable (requires Ampere or newer)")
        return False

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    _TF32_APPLIED = True
    if verbose:
        print("TF32: enabled (matmul + cuDNN); set RETFOUND_TF32=0 for exact fp32")
    return True
