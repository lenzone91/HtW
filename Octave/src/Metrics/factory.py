from .Loss.factory import build_loss
from .MetricSets.factory import build_ac_video_jepa_metric_set, build_metric_set
from .Metrics.factory import build_metric_from_config, build_metrics


def build_ac_video_jepa_metric_stack(
    metric_set_config: dict | None = None,
    loss_config: dict | None = None,
    encoder_shape: dict | None = None,
    runtime_context: dict | None = None,
    strict: bool = True,
) -> dict:
    """
    Build the metric-set/loss pair used by AcVideoJepa modules.

    This replaces the obsolete AcVideoJepaObjective construction path.
    """
    metric_set = build_ac_video_jepa_metric_set(
        config=metric_set_config,
        encoder_shape=encoder_shape,
        runtime_context=runtime_context,
        strict=strict,
    )

    loss = build_loss(
        loss_config=loss_config,
        runtime_context=runtime_context,
        strict=strict,
    )

    return {
        "metric_set": metric_set,
        "loss": loss,
    }