from pathlib import Path

from lightning import LightningModule
from torch import nn

from .model_loading import load_model_state_dict
from .module_loading import load_module_from_lightning_checkpoint


#############################################
# Loading dispatcher
#############################################

# Config-driven loading of weights into already-built objects. This is a utility
# folder (no registry/factory family).
#
# It does not build models or modules, resume Trainer state, or restore
# optimizer/scheduler states.


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

    check_loading_type(loading_config=loading_config, expected_type="torch_model")

    checkpoint_path = resolve_checkpoint_path(loading_config["checkpoint_path"])

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
    return {
        model_name: load_model_if_needed(
            model=model,
            loading_config=loading_config,
            runtime_context=runtime_context,
        )
        for model_name, model in models.items()
    }


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

    check_loading_type(loading_config=loading_config, expected_type="lightning_module")

    checkpoint_path = resolve_checkpoint_path(loading_config["checkpoint_path"])

    return load_module_from_lightning_checkpoint(
        module=module,
        checkpoint_path=checkpoint_path,
        strict=loading_config.get("strict", True),
        map_location=loading_config.get("map_location", "cpu"),
        state_dict_key=loading_config.get("state_dict_key", "state_dict"),
    )


#############################################
# Helpers
#############################################


def is_loading_enabled(loading_config: dict | None) -> bool:
    """
    Check whether loading is enabled.
    """
    if loading_config is None:
        return False

    return loading_config.get("enabled", False)


def check_loading_type(loading_config: dict, expected_type: str) -> None:
    """
    Check that the loading config targets the expected object type.
    """
    loading_type = loading_config.get("type")

    if loading_type != expected_type:
        raise ValueError(
            f"Invalid loading type: {loading_type}. Expected {expected_type}."
        )


def resolve_checkpoint_path(checkpoint_path: str) -> str:
    """
    Resolve a checkpoint path to an absolute path.

    Runtime-context-rooted relative resolution (resolving against project_root /
    run_dir from runtime_context['paths']) is deferred to the Setup migration
    (Decision 22); paths are expanded/resolved against the working directory.
    """
    if checkpoint_path is None:
        raise ValueError("Loading is enabled, but checkpoint_path is None.")

    return str(Path(checkpoint_path).expanduser().resolve())
