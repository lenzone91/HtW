from pathlib import Path

from lightning.pytorch import LightningModule

from .module_loading import load_module_from_lightning_checkpoint


from lightning.pytorch import LightningModule

from . import module_loading  # noqa: F401
from .configs import DEFAULT_MODULE_LOADING_CONFIG
from .module_loading import LightningModuleLoader
from .registry import LOADING_REGISTRY
from ...Workflow.Factory.builder import RegistryBuilder


class DisabledModuleLoader:
    """
    No-op loading policy.
    """

    def __call__(self, module: LightningModule) -> LightningModule:
        return module


def build_module_loader(
    loading_config: dict | None,
    runtime_context: dict | None = None,
) -> LightningModuleLoader | DisabledModuleLoader:
    if not is_loading_enabled(loading_config):
        return DisabledModuleLoader()

    builder = RegistryBuilder(
        registry=LOADING_REGISTRY,
        type_field="type",
    )

    return builder.build_one(
        config=loading_config,
        runtime_context=runtime_context,
    )


def load_module_if_needed(
    module: LightningModule,
    loading_config: dict | None,
    runtime_context: dict | None = None,
) -> LightningModule:
    loader = build_module_loader(
        loading_config=loading_config,
        runtime_context=runtime_context,
    )

    return loader(module)


def is_loading_enabled(loading_config: dict | None) -> bool:
    if loading_config is None:
        return False

    if not isinstance(loading_config, dict):
        raise TypeError(
            "Loading config must be a dictionary, "
            f"got {type(loading_config).__name__}."
        )

    return loading_config.get("enabled", False)


def build_default_module_loader(
    runtime_context: dict | None = None,
) -> LightningModuleLoader | DisabledModuleLoader:
    return build_module_loader(
        loading_config=DEFAULT_MODULE_LOADING_CONFIG,
        runtime_context=runtime_context,
    )
