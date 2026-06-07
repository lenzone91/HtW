from .waveUnet import WaveUNet
from ..Factory.base import BaseBuilder, BaseBuilderDispatcher
from .configs import DEFAULT_WAVEUNET_CONFIG


########################################
# WaveUNet builder
########################################


class WaveUNetBuilder(BaseBuilder):
    """
    Build a WaveUNet model from config.
    """

    def __init__(self, strict: bool = True):
        super().__init__(
            default_config=DEFAULT_WAVEUNET_CONFIG,
            strict=strict,
            check_default_keys=True,
        )

    def build_from_config(
        self,
        config: dict,
        runtime_context: dict | None = None,
    ) -> WaveUNet:
        return WaveUNet(**config)


#########################
# Builder registry
#########################

MODEL_BUILDERS_REGISTRY = {
    "waveunet": WaveUNetBuilder(),
}


#########################
# Builder dispatcher
#########################

class ModelBuilderDispatcher(BaseBuilderDispatcher):
    """
    Build models from named model configs.

    Expected format:
        model_configs[model_name] = model_config
    """

    def __init__(
        self,
        builder_registry: dict = MODEL_BUILDERS_REGISTRY,
        strict: bool = True,
    ):
        super().__init__(
            default_config={},
            builder_registry=builder_registry,
            strict=strict,
            check_default_keys=False,
        )

    def handle_unknown_object(self, object_name: str) -> None:
        self.handle_error(
            f"Unknown model '{object_name}'. "
            f"Available models are: {sorted(self.builder_registry.keys())}."
        )


#########################
# Build models
#########################

def build_models(
    model_configs: dict,
    runtime_context: dict | None = None,
    strict: bool = True,
) -> dict:
    """
    Build several models from a dictionary of configs.
    """
    dispatcher = ModelBuilderDispatcher(strict=strict)

    return dispatcher(
        config=model_configs,
        runtime_context=runtime_context,
    )