"""Device resolution for the runtime context."""

import torch

VALID_DEVICES = {"cpu", "cuda"}


def resolve_device(device: str = "auto") -> str:
    """
    Resolve a device preference to a concrete device string.

    "auto" -> "cuda" if available else "cpu". An explicit "cuda" with no CUDA
    available is an error (fail loud rather than silently fall back).
    """
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"

    if device not in VALID_DEVICES:
        raise ValueError(
            f"Invalid device '{device}'. Expected 'auto' or one of {sorted(VALID_DEVICES)}."
        )

    if device == "cuda" and not torch.cuda.is_available():
        raise ValueError("device='cuda' requested but CUDA is not available.")

    return device
