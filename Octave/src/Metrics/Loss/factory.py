from . import loss as _loss  # noqa: F401
from .configs import DEFAULT_WEIGHTED_METRIC_LOSS_CONFIG
from .registry import LOSS_REGISTRY
from ...Workflow.Factory.builder import RegistryBuilder


def build_loss(
    loss_config: dict | None = None,
    runtime_context: dict | None = None,
    strict: bool = True,
):
    builder = RegistryBuilder(
        registry=LOSS_REGISTRY,
        strict=strict,
        type_field="loss_type",
    )

    return builder.build_one(
        config=loss_config or DEFAULT_WEIGHTED_METRIC_LOSS_CONFIG,
        runtime_context=runtime_context,
    )
