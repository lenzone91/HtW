from pathlib import Path

from torch import nn
from lightning import LightningModule

from .model_loading import load_model_state_dict
from .module_loading import load_module_from_lightning_checkpoint


#############################################
# Loading factory
#############################################

# Config-driven loading dispatcher.
#
# This file does not:
#   - build models;
#   - build Lightning modules;
#   - handle trainer resume;
#   - restore optimizer or scheduler states.


def load_model_if_needed(
    model: nn.Module,
    loading_config: dict | None,
    runtime_context: dict | None = None,
) -> nn.Module:
    """
    Optionally load weights into an already-built torch model.
    """
    if not is_loading_enabled(loading_config):
        return model

    check_loading_type(
        loading_config=loading_config,
        expected_type="torch_model",
    )

    checkpoint_path = resolve_checkpoint_path(
        checkpoint_path=loading_config["checkpoint_path"],
        runtime_context=runtime_context,
        relative_to=loading_config.get("relative_to", "idea_root"),
    )

    return load_model_state_dict(
        model=model,
        checkpoint_path=checkpoint_path,
        strict=loading_config.get("strict", True),
        map_location=loading_config.get("map_location", "cpu"),
        state_dict_key=loading_config.get("state_dict_key"),
    )



def load_models_if_needed(
    models: dict,
    loading_config: dict | None,
    runtime_context: dict | None = None,
) -> dict:
    """
    Optionally load weights into all built models.
    """
    loaded_models = {}

    for model_name, model in models.items():
        loaded_models[model_name] = load_model_if_needed(
            model=model,
            loading_config=loading_config,
            runtime_context=runtime_context,
        )

    return loaded_models


def load_module_if_needed(
    module: LightningModule,
    loading_config: dict | None,
    runtime_context: dict | None = None,
) -> LightningModule:
    """
    Optionally load weights into an already-built LightningModule.
    """
    if not is_loading_enabled(loading_config):
        return module

    check_loading_type(
        loading_config=loading_config,
        expected_type="lightning_module",
    )

    checkpoint_path = resolve_checkpoint_path(
        checkpoint_path=loading_config["checkpoint_path"],
        runtime_context=runtime_context,
        relative_to=loading_config.get("relative_to", "idea_root"),
    )

    return load_module_from_lightning_checkpoint(
        module=module,
        checkpoint_path=checkpoint_path,
        strict=loading_config.get("strict", True),
        map_location=loading_config.get("map_location", "cpu"),
        state_dict_key=loading_config.get("state_dict_key", "state_dict"),
    )


def is_loading_enabled(loading_config: dict | None) -> bool:
    """
    Check whether loading is enabled.
    """
    if loading_config is None:
        return False

    return loading_config.get("enabled", False)


def check_loading_type(
    loading_config: dict,
    expected_type: str,
) -> None:
    """
    Check that the loading config targets the expected object type.
    """
    loading_type = loading_config.get("type")

    if loading_type != expected_type:
        raise ValueError(
            f"Invalid loading type: {loading_type}. "
            f"Expected {expected_type}."
        )


#############################################
# Checkpoint path resolution
#############################################


def resolve_checkpoint_path(
    checkpoint_path: str,
    runtime_context: dict | None = None,
    relative_to: str = "idea_root",
) -> str:
    """
    Resolve a checkpoint path.

    Absolute paths are used as-is.

    Relative paths are resolved with respect to one path root from
    runtime_context["paths"].
    """
    if checkpoint_path is None:
        raise ValueError("Loading is enabled, but checkpoint_path is None.")

    path = Path(checkpoint_path).expanduser()

    if path.is_absolute():
        return str(path.resolve())

    root = get_path_root(
        runtime_context=runtime_context,
        relative_to=relative_to,
    )

    return str((root / path).resolve())


def get_path_root(
    runtime_context: dict | None,
    relative_to: str,
) -> Path:
    """
    Retrieve the root directory used to resolve a relative checkpoint path.
    """
    if runtime_context is None:
        raise ValueError(
            "Cannot resolve a relative checkpoint path without runtime_context."
        )

    paths_context = runtime_context.get("paths", {})

    if relative_to not in paths_context:
        raise KeyError(
            f"Unknown checkpoint path root: {relative_to}. "
            f"Available roots are: {sorted(paths_context.keys())}."
        )

    return Path(paths_context[relative_to]).expanduser().resolve()