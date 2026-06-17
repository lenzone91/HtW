from copy import deepcopy

from . import loggable_metric_set  # noqa: F401  (registers LoggableMetricSet)
from . import metric_set  # noqa: F401  (registers MetricSet)
from .registry import METRIC_SET_BUILDER


#############################################
# Metric set building wrapper
#############################################

DEFAULT_SET_TYPE = "loggable"


def build_metric_set(
    config: dict,
    runtime_context: dict | None = None,
):
    """
    Build one metric set from a metric-set config (routed by `set_type`).

    When `set_type` is absent it defaults to the generic loggable set.
    """
    config = deepcopy(config)
    config.setdefault("set_type", DEFAULT_SET_TYPE)

    return METRIC_SET_BUILDER.build_one(
        config=config,
        runtime_context=runtime_context,
    )
