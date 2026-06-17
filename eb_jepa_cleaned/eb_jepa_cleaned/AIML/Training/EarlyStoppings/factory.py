from lightning.pytorch.callbacks import EarlyStopping

from . import early_stoppings  # noqa: F401  (registers early stopping callbacks)
from .registry import EARLY_STOPPING_BUILDER


#############################################
# Early stopping building wrapper
#############################################


def build_early_stopping_callbacks(
    early_stopping_configs: dict,
    runtime_context: dict | None = None,
) -> list[EarlyStopping]:
    """
    Build all early stopping callbacks from a named config dictionary.
    """
    callbacks = EARLY_STOPPING_BUILDER.build_named(
        configs=early_stopping_configs,
        runtime_context=runtime_context,
    )

    return list(callbacks.values())
