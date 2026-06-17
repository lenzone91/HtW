from . import loss  # noqa: F401  (registers WeightedMetricLoss)
from .registry import LOSS_BUILDER


#############################################
# Loss building wrapper
#############################################


def build_loss(
    config: dict,
    runtime_context: dict | None = None,
):
    """
    Build one loss object from a loss config (routed by `loss_type`).
    """
    return LOSS_BUILDER.build_one(
        config=config,
        runtime_context=runtime_context,
    )
