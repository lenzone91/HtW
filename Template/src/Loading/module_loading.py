import torch
from lightning import LightningModule


#############################################
# Module loading
#############################################

# Utilities for loading Lightning checkpoint weights into already-built modules.
#
# This file does not:
#   - build Lightning modules;
#   - build models;
#   - handle trainer resume;
#   - restore optimizer or scheduler states.


def load_module_from_lightning_checkpoint(
    module: LightningModule,
    checkpoint_path: str,
    strict: bool = True,
    map_location: str | torch.device | None = "cpu",
    state_dict_key: str = "state_dict",
) -> LightningModule:
    """
    Load Lightning checkpoint weights into an already-built LightningModule.

    Args:
        module:
            Already-built LightningModule.

        checkpoint_path:
            Path to a Lightning checkpoint.

        strict:
            Whether checkpoint keys must exactly match module keys.

        map_location:
            Device mapping used by torch.load.

        state_dict_key:
            Key storing the LightningModule state dict in the checkpoint.

    Returns:
        The input module, updated in-place.
    """
    checkpoint = torch.load(
        checkpoint_path,
        map_location=map_location,
    )

    state_dict = checkpoint[state_dict_key]

    module.load_state_dict(
        state_dict,
        strict=strict,
    )

    return module