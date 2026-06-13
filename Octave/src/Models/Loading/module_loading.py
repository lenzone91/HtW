import torch
from lightning.pytorch import LightningModule


def load_module_from_lightning_checkpoint(
    module: LightningModule,
    checkpoint_path: str,
    strict: bool = True,
    map_location: str | torch.device | None = "cpu",
    state_dict_key: str = "state_dict",
) -> LightningModule:
    checkpoint = torch.load(
        checkpoint_path,
        map_location=map_location,
    )

    if state_dict_key not in checkpoint:
        raise KeyError(
            f"Checkpoint does not contain state_dict_key '{state_dict_key}'. "
            f"Available keys are: {sorted(checkpoint.keys())}."
        )

    module.load_state_dict(
        checkpoint[state_dict_key],
        strict=strict,
    )

    return module
