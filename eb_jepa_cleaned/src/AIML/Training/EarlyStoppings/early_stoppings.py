from lightning.pytorch.callbacks import EarlyStopping

from .registry import EARLY_STOPPING_REGISTRY


#############################################
# Early stopping callbacks (thin EarlyStopping wrappers)
#############################################

# Each object is preceded by its own default config.


DEFAULT_BEST_VALUE_STAGNATION_EARLY_STOPPING_CONFIG = {
    "monitor": "val/loss",
    "mode": "min",
    "patience": 10,
    "min_delta": 0.0,
}


@EARLY_STOPPING_REGISTRY.register_class(
    name="best_value_stagnation",
    default_config={
        "early_stopping_type": "best_value_stagnation",
        **DEFAULT_BEST_VALUE_STAGNATION_EARLY_STOPPING_CONFIG,
    },
    type_field="early_stopping_type",
)
class BestValueStagnationEarlyStopping(EarlyStopping):
    pass


DEFAULT_THRESHOLD_EARLY_STOPPING_CONFIG = {
    "monitor": "val/loss",
    "mode": "min",
    "stopping_threshold": 0.01,
}


@EARLY_STOPPING_REGISTRY.register_class(
    name="threshold",
    default_config={
        "early_stopping_type": "threshold",
        **DEFAULT_THRESHOLD_EARLY_STOPPING_CONFIG,
    },
    type_field="early_stopping_type",
)
class ThresholdEarlyStopping(EarlyStopping):
    def __init__(self, **config):
        config = dict(config)
        config.pop("patience", None)
        super().__init__(patience=0, **config)


DEFAULT_DIVERGENCE_EARLY_STOPPING_CONFIG = {
    "monitor": "val/loss",
    "mode": "min",
    "divergence_threshold": 1e3,
    "patience": 0,
}


@EARLY_STOPPING_REGISTRY.register_class(
    name="divergence",
    default_config={
        "early_stopping_type": "divergence",
        **DEFAULT_DIVERGENCE_EARLY_STOPPING_CONFIG,
    },
    type_field="early_stopping_type",
)
class DivergenceEarlyStopping(EarlyStopping):
    def __init__(self, **config):
        config = dict(config)
        config.pop("patience", None)
        super().__init__(patience=0, **config)


DEFAULT_FINITE_VALUE_EARLY_STOPPING_CONFIG = {
    "monitor": "val/loss",
    "mode": "min",
}


@EARLY_STOPPING_REGISTRY.register_class(
    name="finite_value",
    default_config={
        "early_stopping_type": "finite_value",
        **DEFAULT_FINITE_VALUE_EARLY_STOPPING_CONFIG,
    },
    type_field="early_stopping_type",
)
class FiniteValueEarlyStopping(EarlyStopping):
    def __init__(self, **config):
        config = dict(config)
        config.pop("patience", None)
        super().__init__(patience=0, **config)
