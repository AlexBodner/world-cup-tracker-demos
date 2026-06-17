"""Default Torch device selection for local demos."""

from __future__ import annotations


def default_torch_device() -> str:
    """Prefer Apple MPS, then CUDA, else CPU."""
    try:
        import torch
    except ImportError:
        return "cpu"
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"
