from lightning.pytorch.callbacks import EarlyStopping

from ..Factory.base import BaseBuilder, BaseBuilderDispatcher
from .configs import (
    DEFAULT_BEST_VALUE_STAGNATION_EARLY_STOPPING_CONFIG,
    DEFAULT_THRESHOLD_EARLY_STOPPING_CONFIG,
    DEFAULT_DIVERGENCE_EARLY_STOPPING_CONFIG,
    DEFAULT_FINITE_VALUE_EARLY_STOPPING_CONFIG,
    DEFAULT_EARLY_STOPPING_CONFIGS,
)


############################################################
# Stagnation
############################################################


class BestValueStagnationEarlyStoppingBuilder(BaseBuilder):
    """
    Build an early stopping callback based on stagnation of one logged value.
    """

    def __init__(self, strict: bool = True):
        super().__init__(
            default_config=DEFAULT_BEST_VALUE_STAGNATION_EARLY_STOPPING_CONFIG,
            strict=strict,
            check_default_keys=True,
        )

    def build_from_config(
        self,
        config: dict,
        runtime_context: dict | None = None,
    ) -> EarlyStopping:
        return EarlyStopping(**config)


############################################################
# Threshold
############################################################


class ThresholdEarlyStoppingBuilder(BaseBuilder):
    def __init__(self, strict: bool = True):
        super().__init__(
            default_config=DEFAULT_THRESHOLD_EARLY_STOPPING_CONFIG,
            strict=strict,
            check_default_keys=True,
        )

    def build_from_config(
    self,
        config: dict,
        runtime_context: dict | None = None,
    ) -> EarlyStopping:
        config = dict(config)
        config.pop("patience", None)

        return EarlyStopping(
            patience=0,
            **config,
        )


############################################################
# Divergence
############################################################


class DivergenceEarlyStoppingBuilder(BaseBuilder):
    def __init__(self, strict: bool = True):
        super().__init__(
            default_config=DEFAULT_DIVERGENCE_EARLY_STOPPING_CONFIG,
            strict=strict,
            check_default_keys=True,
        )

    def build_from_config(
        self,
        config: dict,
        runtime_context: dict | None = None,
    ) -> EarlyStopping:
        config = dict(config)
        config.pop("patience", None)

        return EarlyStopping(
            patience=0,
            **config,
        )


############################################################
# Finite value
############################################################


class FiniteValueEarlyStoppingBuilder(BaseBuilder):
    def __init__(self, strict: bool = True):
        super().__init__(
            default_config=DEFAULT_FINITE_VALUE_EARLY_STOPPING_CONFIG,
            strict=strict,
            check_default_keys=True,
        )

    def build_from_config(
        self,
        config: dict,
        runtime_context: dict | None = None,
    ) -> EarlyStopping:
        config = dict(config)
        config.pop("patience", None)

        return EarlyStopping(
            patience=0,
            **config,
        )


#####################################################################
# Builder registry
#####################################################################


EARLY_STOPPING_BUILDERS_REGISTRY = {
    "best_value_stagnation": BestValueStagnationEarlyStoppingBuilder(),
    "threshold": ThresholdEarlyStoppingBuilder(),
    "divergence": DivergenceEarlyStoppingBuilder(),
    "finite_value": FiniteValueEarlyStoppingBuilder(),
}


######################################################################
# Dispatcher
######################################################################


class EarlyStoppingBuilderDispatcher(BaseBuilderDispatcher):
    """
    Build early stopping callbacks from named early stopping configs.

    Expected format:
        early_stopping_configs[name] = {
            "early_stopping_type": "...",
            ...
        }
    """

    def __init__(
        self,
        builder_registry: dict = EARLY_STOPPING_BUILDERS_REGISTRY,
        strict: bool = True,
    ):
        super().__init__(
            default_config=DEFAULT_EARLY_STOPPING_CONFIGS,
            builder_registry=builder_registry,
            strict=strict,
            check_default_keys=False,
            check_registered_names=False
        )

    def build_from_config(
        self,
        config: dict,
        runtime_context: dict | None = None,
    ) -> list[EarlyStopping]:
        callbacks = []

        for callback_name, callback_config in config.items():
            callback = self.build_one(
                callback_name=callback_name,
                callback_config=callback_config,
                runtime_context=runtime_context,
            )

            if callback is not None:
                callbacks.append(callback)

        return callbacks

    def build_one(
        self,
        callback_name: str,
        callback_config: dict,
        runtime_context: dict | None = None,
    ) -> EarlyStopping | None:
        if not self.check_config_is_dict(callback_config):
            return None

        callback_config = self.copy_config(callback_config)

        if "early_stopping_type" not in callback_config:
            self.handle_error(
                f"Missing 'early_stopping_type' in early stopping config '{callback_name}'."
            )
            return None

        early_stopping_type = callback_config.pop("early_stopping_type")

        if early_stopping_type not in self.builder_registry:
            self.handle_error(
                f"Unknown early stopping type '{early_stopping_type}' "
                f"for callback '{callback_name}'. "
                f"Available types are: {sorted(self.builder_registry.keys())}."
            )
            return None

        builder = self.builder_registry[early_stopping_type]

        return builder(
            config=callback_config,
            runtime_context=runtime_context,
        )


######################################################################
# General builder
######################################################################


def build_early_stopping_callbacks(
    early_stopping_configs: dict,
    runtime_context: dict | None = None,
    strict: bool = True,
) -> list[EarlyStopping]:
    """
    Build all early stopping callbacks from a dictionary of configs.
    """
    dispatcher = EarlyStoppingBuilderDispatcher(strict=strict)

    return dispatcher(
        config=early_stopping_configs,
        runtime_context=runtime_context,
    )