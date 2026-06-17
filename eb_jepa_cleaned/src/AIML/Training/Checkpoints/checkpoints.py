from lightning.pytorch.callbacks import ModelCheckpoint

from .registry import CHECKPOINT_REGISTRY


#############################################
# Checkpoint callbacks (thin ModelCheckpoint wrappers)
#############################################

# Each object is preceded by its own default config. dirpath defaults to None:
# with no Setup runtime-context path resolution yet (Decision 22), Lightning's
# default checkpoint directory is used unless an explicit dirpath is in config.


DEFAULT_LAST_CHECKPOINT_CONFIG = {
    "dirpath": None,
    "filename": "last",
    "every_n_epochs": 1,
}


@CHECKPOINT_REGISTRY.register_class(
    name="last",
    default_config={"checkpoint_type": "last", **DEFAULT_LAST_CHECKPOINT_CONFIG},
    type_field="checkpoint_type",
)
class LastCheckpoint(ModelCheckpoint):
    def __init__(self, **config):
        super().__init__(save_last=True, save_top_k=0, **config)


DEFAULT_PERIODIC_CHECKPOINT_CONFIG = {
    "dirpath": None,
    "filename": "epoch={epoch}-step={step}",
    "every_n_epochs": 1,
}


@CHECKPOINT_REGISTRY.register_class(
    name="periodic",
    default_config={
        "checkpoint_type": "periodic",
        **DEFAULT_PERIODIC_CHECKPOINT_CONFIG,
    },
    type_field="checkpoint_type",
)
class PeriodicCheckpoint(ModelCheckpoint):
    def __init__(self, **config):
        super().__init__(save_last=False, save_top_k=-1, **config)


DEFAULT_BEST_VALUE_CHECKPOINT_CONFIG = {
    "dirpath": None,
    "filename": "best-{epoch}-{step}-{val_loss:.4f}",
    "monitor": "val/loss",
    "mode": "min",
    "save_top_k": 1,
}


@CHECKPOINT_REGISTRY.register_class(
    name="best_value",
    default_config={
        "checkpoint_type": "best_value",
        **DEFAULT_BEST_VALUE_CHECKPOINT_CONFIG,
    },
    type_field="checkpoint_type",
)
class BestValueCheckpoint(ModelCheckpoint):
    def __init__(self, **config):
        super().__init__(save_last=False, **config)
