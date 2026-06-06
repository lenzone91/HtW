import torch
from torch import nn


#############################################
# Model loading
#############################################

# Utilities for loading torch-based weights into already-built models.
#
# This file does not:
#   - build model architectures;
#   - build Lightning modules;
#   - handle trainer resume;
#   - load optimizer or scheduler states.


def load_model_state_dict(
    model: nn.Module,
    checkpoint_path: str,
    strict: bool = True,
    map_location: str | torch.device | None = "cpu",
    state_dict_key: str | None = None,
) -> nn.Module:
    """
    Load torch weights into an already-built model.

    Args:
        model:
            Already-built torch module.

        checkpoint_path:
            Path to a torch checkpoint.

        strict:
            Whether loaded keys must exactly match model keys.

        map_location:
            Device mapping used by torch.load.

        state_dict_key:
            Optional key used when the checkpoint stores the state dict inside
            a larger dictionary.

    Returns:
        The input model, updated in-place.
    """
    checkpoint = torch.load(
        checkpoint_path,
        map_location=map_location,
    )

    state_dict = extract_state_dict(
        checkpoint=checkpoint,
        state_dict_key=state_dict_key,
    )

    model.load_state_dict(
        state_dict,
        strict=strict,
    )

    return model


def extract_state_dict(
    checkpoint,
    state_dict_key: str | None = None,
) -> dict:
    """
    Extract a state dict from a torch checkpoint.
    """
    if state_dict_key is not None:
        return checkpoint[state_dict_key]

    if is_state_dict(checkpoint):
        return checkpoint

    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        return checkpoint["state_dict"]

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        return checkpoint["model_state_dict"]

    raise ValueError(
        "Could not infer model state dict from checkpoint. "
        "Provide state_dict_key explicitly."
    )


def is_state_dict(value) -> bool:
    """
    Check whether a value looks like a torch state dict.
    """
    if not isinstance(value, dict):
        return False

    if len(value) == 0:
        return False

    return all(
        isinstance(key, str)
        for key in value.keys()
    )