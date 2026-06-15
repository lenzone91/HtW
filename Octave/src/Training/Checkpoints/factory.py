from . import checkpoints  # noqa: F401
from .base import NamedModelCheckpoint
from .configs import DEFAULT_CHECKPOINT_CONFIGS
from ...Workflow.Factory.builder import RegistryBuilder
from .registry import CHECKPOINT_REGISTRY


def build_checkpoint_callbacks(
    checkpoint_configs: dict | None = None,
    runtime_context: dict | None = None,
    strict: bool = True,
) -> list[NamedModelCheckpoint]:
    if checkpoint_configs is None:
        checkpoint_configs = DEFAULT_CHECKPOINT_CONFIGS

    if checkpoint_configs == {}:
        return []

    if not isinstance(checkpoint_configs, dict):
        raise TypeError(
            "Checkpoint configs must be a dictionary, "
            f"got {type(checkpoint_configs).__name__}."
        )

    callbacks = []

    for checkpoint_name, checkpoint_config in checkpoint_configs.items():
        callbacks.append(
            build_checkpoint_callback(
                checkpoint_name=checkpoint_name,
                checkpoint_config=checkpoint_config,
                runtime_context=runtime_context,
                strict=strict,
            )
        )

    return callbacks


def build_checkpoint_callback(
    checkpoint_name: str,
    checkpoint_config: dict,
    runtime_context: dict | None = None,
    strict: bool = True,
) -> NamedModelCheckpoint:
    if not isinstance(checkpoint_name, str) or checkpoint_name.strip() == "":
        raise TypeError(
            "Checkpoint name must be a non-empty string, "
            f"got {checkpoint_name!r}."
        )

    builder = RegistryBuilder(
        registry=CHECKPOINT_REGISTRY,
        strict=strict,
        type_field="checkpoint_type",
    )

    return builder.build_one(
        config={
            **checkpoint_config,
            "checkpoint_name": checkpoint_name,
        },
        runtime_context=runtime_context,
    )
