from pathlib import Path

from lightning.pytorch.callbacks import ModelCheckpoint

from ..Factory.base import BaseBuilder, BaseBuilderDispatcher
from .configs import (
    DEFAULT_LAST_CHECKPOINT_CONFIG,
    DEFAULT_PERIODIC_CHECKPOINT_CONFIG,
    DEFAULT_BEST_VALUE_CHECKPOINT_CONFIG,
    DEFAULT_CHECKPOINT_CONFIGS,
)


###############################
# Save last checkpoint builder
###############################


class LastCheckpointBuilder(BaseBuilder):
    def __init__(self, strict: bool = True):
        super().__init__(
            default_config=DEFAULT_LAST_CHECKPOINT_CONFIG,
            strict=strict,
            check_default_keys=True,
        )

    def build_from_config(
        self,
        config: dict,
        runtime_context: dict | None = None,
    ) -> ModelCheckpoint:
        config = resolve_checkpoint_dirpath(
            config=config,
            runtime_context=runtime_context,
        )

        return ModelCheckpoint(
            save_last=True,
            save_top_k=0,
            **config,
        )


########################################
# Save periodically checkpoint builder
########################################


class PeriodicCheckpointBuilder(BaseBuilder):
    def __init__(self, strict: bool = True):
        super().__init__(
            default_config=DEFAULT_PERIODIC_CHECKPOINT_CONFIG,
            strict=strict,
            check_default_keys=True,
        )

    def build_from_config(
        self,
        config: dict,
        runtime_context: dict | None = None,
    ) -> ModelCheckpoint:
        config = resolve_checkpoint_dirpath(
            config=config,
            runtime_context=runtime_context,
        )

        return ModelCheckpoint(
            save_last=False,
            save_top_k=-1,
            **config,
        )


########################################
# Save best checkpoint builder
########################################


class BestValueCheckpointBuilder(BaseBuilder):
    def __init__(self, strict: bool = True):
        super().__init__(
            default_config=DEFAULT_BEST_VALUE_CHECKPOINT_CONFIG,
            strict=strict,
            check_default_keys=True,
        )

    def build_from_config(
        self,
        config: dict,
        runtime_context: dict | None = None,
    ) -> ModelCheckpoint:
        config = resolve_checkpoint_dirpath(
            config=config,
            runtime_context=runtime_context,
        )

        return ModelCheckpoint(
            save_last=False,
            **config,
        )


########################
# Path helpers
########################


def resolve_checkpoint_dirpath(
    config: dict,
    runtime_context: dict | None = None,
) -> dict:
    """
    Resolve checkpoint callback dirpath.

    If dirpath is None, use runtime_context["paths"]["checkpoints_dir"].
    If dirpath is absolute, use it directly.
    If dirpath is relative, resolve it against run_dir.
    """
    config = dict(config)

    dirpath = config.get("dirpath")

    if dirpath is None:
        config["dirpath"] = get_default_checkpoint_dir(runtime_context)
        return config

    path = Path(dirpath).expanduser()

    if path.is_absolute():
        config["dirpath"] = str(path.resolve())
        return config

    root = get_path_root(
        runtime_context=runtime_context,
        root_key="run_dir",
    )

    config["dirpath"] = str((root / path).resolve())
    return config


def get_default_checkpoint_dir(
    runtime_context: dict | None = None,
) -> str | None:
    """
    Return the default checkpoint directory from runtime_context.

    If runtime_context is absent, keep Lightning's default behavior.
    """
    if runtime_context is None:
        return None

    paths_context = runtime_context.get("paths", {})

    if "checkpoints_dir" not in paths_context:
        raise KeyError(
            "runtime_context['paths'] does not contain 'checkpoints_dir'."
        )

    return paths_context["checkpoints_dir"]


def get_path_root(
    runtime_context: dict | None,
    root_key: str,
) -> Path:
    """
    Retrieve one root directory from runtime_context["paths"].
    """
    if runtime_context is None:
        raise ValueError(
            f"Cannot resolve relative checkpoint dirpath without runtime_context."
        )

    paths_context = runtime_context.get("paths", {})

    if root_key not in paths_context:
        raise KeyError(
            f"Unknown checkpoint path root: {root_key}. "
            f"Available roots are: {sorted(paths_context.keys())}."
        )

    return Path(paths_context[root_key]).expanduser().resolve()


########################
# Builders registry
########################


CHECKPOINT_BUILDERS_REGISTRY = {
    "last": LastCheckpointBuilder(),
    "periodic": PeriodicCheckpointBuilder(),
    "best_value": BestValueCheckpointBuilder(),
}


##########################
# Builder dispatcher
##########################


class CheckpointBuilderDispatcher(BaseBuilderDispatcher):
    """
    Build checkpoint callbacks from named checkpoint configs.

    Unlike most project dispatchers, the top-level config keys are checkpoint
    instance names, not checkpoint builder names.

    Example:
        "best_val_loss": {
            "checkpoint_type": "best_value",
            ...
        }

    Therefore, default-key checking against the builder registry must be disabled:
    "best_val_loss" is a user-facing callback name, while "best_value" is the
    registered builder type.
    """

    def __init__(
        self,
        builder_registry: dict = CHECKPOINT_BUILDERS_REGISTRY,
        strict: bool = True,
    ):
        super().__init__(
            default_config=DEFAULT_CHECKPOINT_CONFIGS,
            builder_registry=builder_registry,
            strict=strict,
            check_default_keys=False,
            check_registered_names=False
        )

    def build_from_config(
        self,
        config: dict,
        runtime_context: dict | None = None,
    ) -> list[ModelCheckpoint]:
        callbacks = []

        for checkpoint_name, checkpoint_config in config.items():
            callback = self.build_one(
                checkpoint_name=checkpoint_name,
                checkpoint_config=checkpoint_config,
                runtime_context=runtime_context,
            )

            if callback is not None:
                callbacks.append(callback)

        return callbacks

    def build_one(
        self,
        checkpoint_name: str,
        checkpoint_config: dict,
        runtime_context: dict | None = None,
    ) -> ModelCheckpoint | None:
        if not self.check_config_is_dict(checkpoint_config):
            return None

        checkpoint_config = self.copy_config(checkpoint_config)

        if "checkpoint_type" not in checkpoint_config:
            self.handle_error(
                f"Missing 'checkpoint_type' in checkpoint config '{checkpoint_name}'."
            )
            return None

        checkpoint_type = checkpoint_config.pop("checkpoint_type")

        if checkpoint_type not in self.builder_registry:
            self.handle_unknown_checkpoint_type(
                checkpoint_name=checkpoint_name,
                checkpoint_type=checkpoint_type,
            )
            return None

        builder = self.builder_registry[checkpoint_type]

        return builder(
            config=checkpoint_config,
            runtime_context=runtime_context,
        )

    def handle_unknown_checkpoint_type(
        self,
        checkpoint_name: str,
        checkpoint_type: str,
    ) -> None:
        self.handle_error(
            f"Unknown checkpoint type '{checkpoint_type}' "
            f"for checkpoint '{checkpoint_name}'. "
            f"Available checkpoint types are: {sorted(self.builder_registry.keys())}."
        )


########################
# Build wrapper
########################


def build_checkpoint_callbacks(
    checkpoint_configs: dict,
    runtime_context: dict | None = None,
    strict: bool = True,
) -> list[ModelCheckpoint]:
    """
    Build all checkpoint callbacks from a dictionary of configs.
    """
    dispatcher = CheckpointBuilderDispatcher(strict=strict)

    return dispatcher(
        config=checkpoint_configs,
        runtime_context=runtime_context,
    )