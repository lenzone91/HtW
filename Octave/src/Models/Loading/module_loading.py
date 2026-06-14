from pathlib import Path

import torch
from lightning.pytorch import LightningModule

from .configs import DEFAULT_MODULE_LOADING_CONFIG
from .registry import LOADING_REGISTRY
from ...Workflow.Factory.registry import FieldResolution


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


def resolve_checkpoint_path(
    config: dict,
    runtime_context: dict | None = None,
    strict: bool = True,
    **kwargs,
) -> str:
    checkpoint_path = config.get("checkpoint_path")

    if checkpoint_path is None:
        raise ValueError("Loading is enabled, but checkpoint_path is None.")

    path = Path(checkpoint_path).expanduser()

    if path.is_absolute():
        return str(path.resolve())

    relative_to = config.get("relative_to", "run_dir")
    root = get_path_root(
        runtime_context=runtime_context,
        relative_to=relative_to,
    )

    return str((root / path).resolve())


def get_path_root(
    runtime_context: dict | None,
    relative_to: str,
) -> Path:
    if runtime_context is None:
        raise ValueError(
            "Cannot resolve relative checkpoint path without runtime_context."
        )

    paths_context = runtime_context.get("paths", {})

    if relative_to not in paths_context:
        raise KeyError(
            f"Unknown checkpoint path root: {relative_to}. "
            f"Available roots are: {sorted(paths_context.keys())}."
        )

    return Path(paths_context[relative_to]).expanduser().resolve()


@LOADING_REGISTRY.register_class(
    name="lightning_module",
    default_config=DEFAULT_MODULE_LOADING_CONFIG,
    type_field="type",
    field_resolutions=(
        FieldResolution(
            target_key="checkpoint_path",
            resolver=resolve_checkpoint_path,
        ),
    ),
)
class LightningModuleLoader:
    """
    Callable weight-loading policy for an already-built LightningModule.
    """

    def __init__(
        self,
        enabled: bool = True,
        checkpoint_path: str | None = None,
        strict: bool = True,
        map_location: str | torch.device | None = "cpu",
        state_dict_key: str = "state_dict",
        relative_to: str = "run_dir",
    ) -> None:
        if not enabled:
            raise ValueError(
                "LightningModuleLoader should only be built for enabled loading configs."
            )

        self.checkpoint_path = checkpoint_path
        self.strict = strict
        self.map_location = map_location
        self.state_dict_key = state_dict_key
        self.relative_to = relative_to

    def __call__(self, module: LightningModule) -> LightningModule:
        return load_module_from_lightning_checkpoint(
            module=module,
            checkpoint_path=self.checkpoint_path,
            strict=self.strict,
            map_location=self.map_location,
            state_dict_key=self.state_dict_key,
        )
