from . import metric_set  # noqa: F401
from .configs import DEFAULT_AC_VIDEO_JEPA_METRIC_SET_CONFIG
from .registry import METRIC_SET_REGISTRY
from ...Workflow.Factory.builder import RegistryBuilder


def build_metric_set(
    metric_set_config: dict | None = None,
    encoder_shape: dict | None = None,
    runtime_context: dict | None = None,
    strict: bool = True,
):
    builder = RegistryBuilder(
        registry=METRIC_SET_REGISTRY,
        strict=strict,
        type_field="set_type",
    )

    return builder.build_one(
        config=metric_set_config or DEFAULT_AC_VIDEO_JEPA_METRIC_SET_CONFIG,
        runtime_context=runtime_context,
        encoder_shape=encoder_shape,
    )


def build_ac_video_jepa_metric_set(
    config: dict | None = None,
    encoder_shape: dict | None = None,
    runtime_context: dict | None = None,
    strict: bool = True,
):
    return build_metric_set(
        metric_set_config=config or DEFAULT_AC_VIDEO_JEPA_METRIC_SET_CONFIG,
        encoder_shape=encoder_shape,
        runtime_context=runtime_context,
        strict=strict,
    )