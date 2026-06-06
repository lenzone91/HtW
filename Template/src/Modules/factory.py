from .waveUnet import WaveUNetLightningModule

from ..Factory.base import BaseBuilder, BaseBuilderDispatcher
from ..Models.factory import build_models
from ..Metrics.factory import build_metric_set, build_loss
from ..Loading.factory import load_models_if_needed

from .configs import DEFAULT_WAVEUNET_LIGHTNING_MODULE_CONFIG


###############################
# Builder
###############################


class WaveUNetLightningModuleBuilder(BaseBuilder):
    """
    Build a WaveUNet LightningModule from a full module config.
    """

    def __init__(self, strict: bool = True):
        super().__init__(
            default_config=DEFAULT_WAVEUNET_LIGHTNING_MODULE_CONFIG,
            strict=strict,
            check_default_keys=True,
        )

    def build_from_config(
        self,
        config: dict,
        runtime_context: dict | None = None,
        loading_config: dict | None = None,
    ) -> WaveUNetLightningModule:
        models = build_models(
            model_configs=config["model_configs"],
            runtime_context=runtime_context,
        )

        model_loading_config = get_model_loading_config(loading_config)

        models = load_models_if_needed(
            models=models,
            loading_config=model_loading_config,
            runtime_context=runtime_context,
        )

        train_metrics = build_metric_set(
            config=config["train_metrics_config"],
            runtime_context=runtime_context,
        )

        val_metrics = build_metric_set(
            config=config["val_metrics_config"],
            runtime_context=runtime_context,
        )

        test_metrics = build_metric_set(
            config=config["test_metrics_config"],
            runtime_context=runtime_context,
        )

        loss = build_loss(
            config=config["loss_weights"],
            runtime_context=runtime_context,
        )


        return WaveUNetLightningModule(
            **models,
            train_metrics=train_metrics,
            val_metrics=val_metrics,
            test_metrics=test_metrics,
            loss=loss,
            optimizer_configs=config["optimizer_configs"],
            scheduler_configs=config.get("scheduler_configs", {}),
            log_loss_ml_steps=config["log_loss_ml_steps"],
        )


################################
# Builder registry
################################


LIGHTNING_MODULE_BUILDERS_REGISTRY = {
    "waveunet": WaveUNetLightningModuleBuilder(),
}


################################
# Builder dispatcher
################################


class LightningModuleBuilderDispatcher(BaseBuilderDispatcher):
    """
    Build exactly one LightningModule.
    """

    def __init__(
        self,
        builder_registry: dict = LIGHTNING_MODULE_BUILDERS_REGISTRY,
        strict: bool = True,
    ):
        super().__init__(
            default_config={},
            builder_registry=builder_registry,
            strict=strict,
            check_default_keys=False,
        )

    def __call__(
        self,
        config: dict,
        runtime_context: dict | None = None,
        *args,
        **kwargs,
    ):
        modules = super().__call__(
            config=config,
            runtime_context=runtime_context,
            *args,
            **kwargs,
        )

        if len(modules) != 1:
            self.handle_error(
                f"Expected exactly one LightningModule, got {len(modules)}."
            )
            return None

        module_name = list(modules.keys())[0]
        return modules[module_name]


#############################
# Loading helpers
#############################


def get_model_loading_config(
    loading_config: dict | None,
) -> dict | None:
    """
    Extract model loading config from the global loading config.
    """
    if loading_config is None:
        return None

    return loading_config.get("model")





#############################
# Wrapper
#############################


def build_lightning_module(
    lightning_module_configs: dict,
    runtime_context: dict | None = None,
    loading_config: dict | None = None,
    strict: bool = True,
):
    """
    Build exactly one LightningModule from named module configs.
    """
    dispatcher = LightningModuleBuilderDispatcher(strict=strict)

    return dispatcher(
        config=lightning_module_configs,
        runtime_context=runtime_context,
        loading_config=loading_config,
    )