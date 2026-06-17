from .registry import LIGHTNING_MODULE_BUILDER


#############################################
# Lightning module building wrappers
#############################################

# Thin entrypoints over the shared module builder. Concrete modules register
# themselves in their own object files (audio/TSE, Phase 3), so none are imported
# here yet.
#
# Construction and weight loading are separate concerns: these wrappers build the
# module; checkpoint loading is applied afterwards via Models/Loading.


def build_lightning_modules(
    lightning_module_configs: dict,
    runtime_context: dict | None = None,
) -> dict:
    """
    Build named LightningModules from a named module config dictionary.
    """
    return LIGHTNING_MODULE_BUILDER.build_named(
        configs=lightning_module_configs,
        runtime_context=runtime_context,
    )


def build_lightning_module(
    lightning_module_configs: dict,
    runtime_context: dict | None = None,
):
    """
    Build exactly one LightningModule from a named module config dictionary.
    """
    modules = build_lightning_modules(
        lightning_module_configs=lightning_module_configs,
        runtime_context=runtime_context,
    )

    if len(modules) != 1:
        raise ValueError(
            f"Expected exactly one LightningModule, got {len(modules)}."
        )

    return next(iter(modules.values()))
