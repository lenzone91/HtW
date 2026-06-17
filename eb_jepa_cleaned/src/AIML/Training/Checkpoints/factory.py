from lightning.pytorch.callbacks import ModelCheckpoint

from . import checkpoints  # noqa: F401  (registers checkpoint callbacks)
from .registry import CHECKPOINT_BUILDER


#############################################
# Checkpoint building wrapper
#############################################

# Resolution of dirpath against the runtime-context paths (run_dir /
# checkpoints_dir) is deferred to the Setup migration (Decision 22); dirpath is
# taken from config as-is for now.


def build_checkpoint_callbacks(
    checkpoint_configs: dict,
    runtime_context: dict | None = None,
) -> list[ModelCheckpoint]:
    """
    Build all checkpoint callbacks from a named config dictionary.
    """
    callbacks = CHECKPOINT_BUILDER.build_named(
        configs=checkpoint_configs,
        runtime_context=runtime_context,
    )

    return list(callbacks.values())
