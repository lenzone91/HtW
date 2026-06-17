from .registry import METRIC_BUILDER


#############################################
# Metric building wrappers
#############################################

# Thin entrypoints over the shared metric builder. Concrete metrics register
# themselves in their own object files. All current metrics are audio-specific
# (Phase 3), so none are imported here yet.


def build_metric(
    metric_config: dict,
    metric_name: str,
    runtime_context: dict | None = None,
):
    """
    Build one metric from one config and an explicit registered name.
    """
    return METRIC_BUILDER.build_one(
        config=metric_config,
        runtime_context=runtime_context,
        name=metric_name,
    )


def build_metrics(
    metric_configs: dict,
    runtime_context: dict | None = None,
) -> dict:
    """
    Build several metrics from a named metric config dictionary.
    """
    return METRIC_BUILDER.build_named(
        configs=metric_configs,
        runtime_context=runtime_context,
    )
