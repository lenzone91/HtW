from torch import nn

from .registry import METRIC_SET_REGISTRY, METRICS_SUB_BUILD


#############################################
# Metric set
#############################################

DEFAULT_METRIC_SET_CONFIG = {
    "set_type": "metric",
    "metrics": {},
}


@METRIC_SET_REGISTRY.register_class(
    name="metric",
    default_config=DEFAULT_METRIC_SET_CONFIG,
    type_field="set_type",
    sub_builds=(METRICS_SUB_BUILD,),
)
class MetricSet(nn.Module):
    """
    Collection of metric modules.

    General-purpose, modality-agnostic metric orchestration:
        - stores named metrics (validated as nn.Module);
        - evaluates each metric on its own inputs.

    Input dispatch convention (per metric):
        tensor       -> metric(x)
        tuple/list   -> metric(*inputs)
        dict         -> metric(**inputs)

    Strict: evaluating an unregistered metric name raises. This class knows
    nothing about Lightning logging or train/val/test phases.
    """

    def __init__(
        self,
        metrics: dict[str, nn.Module] | None = None,
        **metric_kwargs: nn.Module,
    ):
        super().__init__()

        metrics = self.prepare_metrics(metrics, **metric_kwargs)
        self.check_metrics(metrics)

        # ModuleDict registers metric modules properly.
        self.metrics = nn.ModuleDict(metrics)

    def prepare_metrics(
        self,
        metrics: dict[str, nn.Module] | None = None,
        **metric_kwargs: nn.Module,
    ) -> dict[str, nn.Module]:
        """
        Merge metrics given as a dict and as keyword arguments.
        """
        if metrics is None:
            metrics = {}

        return {**metrics, **metric_kwargs}

    def forward(self, metrics_inputs: dict) -> dict:
        """
        Evaluate all requested metrics. `metrics_inputs[name]` is the input
        specification for that metric.
        """
        return self.evaluate_metrics(metrics_inputs)

    def evaluate_metrics(self, metrics_inputs: dict) -> dict:
        return {
            metric_name: self.evaluate_metric(metric_name, metric_inputs)
            for metric_name, metric_inputs in metrics_inputs.items()
        }

    def evaluate_metric(self, metric_name: str, metric_inputs):
        if metric_name not in self.metrics:
            raise KeyError(
                f"'{metric_name}' is not registered in this MetricSet. "
                f"Available metrics: {sorted(self.metrics.keys())}."
            )

        return self.call_metric(self.metrics[metric_name], metric_inputs)

    def call_metric(self, metric: nn.Module, metric_inputs):
        if isinstance(metric_inputs, dict):
            return metric(**metric_inputs)

        if isinstance(metric_inputs, (tuple, list)):
            return metric(*metric_inputs)

        return metric(metric_inputs)

    def check_metrics(self, metrics: dict) -> None:
        for metric_name, metric in metrics.items():
            if not isinstance(metric, nn.Module):
                raise TypeError(
                    f"Metric {metric_name} must be an nn.Module, got {type(metric)}."
                )

    def __str__(self) -> str:
        metric_names = "\n".join(self.metrics.keys())
        return f"MetricSet with:\n{metric_names}"
