from .configs import DEFAULT_METRIC_CONFIGS
from .registry import METRIC_REGISTRY
from ...Workflow.Factory.builder import RegistryBuilder


def build_metric_from_config(
    metric_config: dict,
    metric_name: str,
    encoder_shape: dict | None = None,
    runtime_context: dict | None = None,
    strict: bool = True,
):
    builder = RegistryBuilder(
        registry=METRIC_REGISTRY,
        strict=strict,
        type_field="metric_type",
    )

    return builder.build_one(
        config=metric_config,
        runtime_context=runtime_context,
        name=metric_name,
        encoder_shape=encoder_shape,
    )


def build_metrics(
    metric_configs: dict | None = None,
    encoder_shape: dict | None = None,
    runtime_context: dict | None = None,
    strict: bool = True,
) -> dict:
    builder = RegistryBuilder(
        registry=METRIC_REGISTRY,
        strict=strict,
        type_field="metric_type",
    )

    return builder.build_named(
        configs=metric_configs or DEFAULT_METRIC_CONFIGS,
        runtime_context=runtime_context,
        encoder_shape=encoder_shape,
    )