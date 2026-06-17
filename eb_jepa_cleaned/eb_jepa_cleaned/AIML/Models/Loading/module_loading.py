import torch
from lightning import LightningModule


#############################################
# Module loading
#############################################

# Utilities for loading Lightning checkpoint weights into already-built modules.
#
# This file does not build modules, build models, handle trainer resume, or
# restore optimizer/scheduler states.


def load_module_from_lightning_checkpoint(
    module: LightningModule,
    checkpoint_path: str,
    strict: bool = True,
    map_location: str | torch.device | None = "cpu",
    state_dict_key: str = "state_dict",
) -> LightningModule:
    """
    Load Lightning checkpoint weights into an already-built LightningModule
    (in place). `strict` is torch's `load_state_dict(strict=...)`.
    """
    checkpoint = torch.load(checkpoint_path, map_location=map_location)

    state_dict = checkpoint[state_dict_key]

    module.load_state_dict(state_dict, strict=strict)

    return module
