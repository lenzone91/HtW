from copy import deepcopy
from pathlib import Path

from lightning.pytorch.callbacks import ModelCheckpoint

from .configs import (
    DEFAULT_BEST_VALUE_CHECKPOINT_CONFIG,
    DEFAULT_LAST_CHECKPOINT_CONFIG,
    DEFAULT_PERIODIC_CHECKPOINT_CONFIG,
)


CHECKPOINT_DEFAULTS_BY_TYPE = {
    "last": DEFAULT_LAST_CHECKPOINT_CONFIG,
    "periodic": DEFAULT_PERIODIC_CHECKPOINT_CONFIG,
    "best_value": DEFAULT_BEST_VALUE_CHECKPOINT_CONFIG,
}


def build_checkpoint_callbacks(
    checkpoint_configs: dict,
    runtime_context: dict | None = None,
) -> list[ModelCheckpoint]:
    if checkpoint_configs == {}:
        return []

    if not isinstance(checkpoint_configs, dict):
        raise TypeError(
            "Checkpoint configs must be a dictionary, "
            f"got {type(checkpoint_configs).__name__}."
        )

    callbacks = []

    for checkpoint_name, checkpoint_config in deepcopy(checkpoint_configs).items():
        callbacks.append(
            build_checkpoint_callback(
                checkpoint_name=checkpoint_name,
                checkpoint_config=checkpoint_config,
                runtime_context=runtime_context,
            )
        )

    return callbacks


def build_checkpoint_callback(
    checkpoint_name: str,
    checkpoint_config: dict,
    runtime_context: dict | None = None,
) -> ModelCheckpoint:
    prepared_config = prepare_checkpoint_config(
        checkpoint_name=checkpoint_name,
        checkpoint_config=checkpoint_config,
        runtime_context=runtime_context,
    )
    checkpoint_type = prepared_config.pop("checkpoint_type")

    if checkpoint_type == "last":
        return ModelCheckpoint(
            save_last=True,
            save_top_k=0,
            **prepared_config,
        )

    if checkpoint_type == "periodic":
        return ModelCheckpoint(
            save_last=False,
            save_top_k=-1,
            **prepared_config,
        )

    if checkpoint_type == "best_value":
        return ModelCheckpoint(
            save_last=False,
            **prepared_config,
        )

    raise KeyError(f"Unknown checkpoint_type '{checkpoint_type}'.")


def prepare_checkpoint_config(
    checkpoint_name: str,
    checkpoint_config: dict,
    runtime_context: dict | None = None,
) -> dict:
    if not isinstance(checkpoint_config, dict):
        raise TypeError(
            f"Checkpoint config '{checkpoint_name}' must be a dictionary, "
            f"got {type(checkpoint_config).__name__}."
        )

    checkpoint_type = checkpoint_config.get("checkpoint_type")
    if checkpoint_type not in CHECKPOINT_DEFAULTS_BY_TYPE:
        raise KeyError(
            f"Unknown checkpoint_type '{checkpoint_type}' for checkpoint "
            f"'{checkpoint_name}'. Available checkpoint types are: "
            f"{sorted(CHECKPOINT_DEFAULTS_BY_TYPE)}."
        )

    default_config = CHECKPOINT_DEFAULTS_BY_TYPE[checkpoint_type]
    prepared_config = deepcopy(default_config)
    user_config = deepcopy(checkpoint_config)

    unknown_keys = set(user_config) - set(default_config)
    if unknown_keys:
        raise KeyError(
            f"Unknown checkpoint config keys for '{checkpoint_name}': "
            f"{sorted(unknown_keys)}. "
            f"Allowed keys are: {sorted(default_config)}."
        )

    prepared_config.update(user_config)
    prepared_config = resolve_checkpoint_dirpath(
        config=prepared_config,
        runtime_context=runtime_context,
    )

    return prepared_config


def resolve_checkpoint_dirpath(
    config: dict,
    runtime_context: dict | None = None,
) -> dict:
    config = dict(config)
    dirpath = config.get("dirpath")

    if dirpath is None:
        config["dirpath"] = get_default_checkpoint_dir(runtime_context)
        return config

    path = Path(dirpath).expanduser()

    if path.is_absolute():
        config["dirpath"] = str(path.resolve())
        return config

    root = get_path_root(runtime_context, root_key="run_dir")
    config["dirpath"] = str((root / path).resolve())
    return config


def get_default_checkpoint_dir(runtime_context: dict | None = None) -> str | None:
    if runtime_context is None:
        return None

    checkpoints_dir = runtime_context.get("paths", {}).get("checkpoints_dir")
    if checkpoints_dir is None:
        raise KeyError("runtime_context['paths']['checkpoints_dir'] is required.")

    return checkpoints_dir


def get_path_root(runtime_context: dict | None, root_key: str) -> Path:
    if runtime_context is None:
        raise ValueError("runtime_context is required to resolve relative dirpath.")

    root = runtime_context.get("paths", {}).get(root_key)
    if root is None:
        raise KeyError(f"runtime_context['paths']['{root_key}'] is required.")

    return Path(root).expanduser().resolve()
