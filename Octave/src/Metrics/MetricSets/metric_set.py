import torch
from torch import nn

from .configs import DEFAULT_AC_VIDEO_JEPA_METRIC_SET_CONFIG
from .registry import METRIC_SET_REGISTRY, METRICS_SUB_BUILD


class MetricSet(nn.Module):
    """
    Collection of independently registered metric modules.
    """

    def __init__(
        self,
        strict: bool = True,
        metrics: dict[str, nn.Module] | None = None,
        **extra_metrics: nn.Module,
    ) -> None:
        super().__init__()

        prepared_metrics = {}

        if metrics is not None:
            prepared_metrics.update(metrics)

        prepared_metrics.update(extra_metrics)

        self.strict = strict
        self.check_metrics(prepared_metrics)
        self.metrics = nn.ModuleDict(prepared_metrics)

    def forward(self, metrics_inputs: dict) -> dict:
        metrics_values = {}

        for metric_name, metric_inputs in metrics_inputs.items():
            metric_value = self.evaluate_metric(metric_name, metric_inputs)
            if metric_value is not None:
                metrics_values[metric_name] = metric_value

        return metrics_values

    def evaluate_metric(self, metric_name: str, metric_inputs):
        if metric_name not in self.metrics:
            self.handle_error(f"{metric_name} is not registered in MetricSet.")
            return None

        metric = self.metrics[metric_name]
        if isinstance(metric_inputs, dict):
            return metric(**metric_inputs)
        if isinstance(metric_inputs, (tuple, list)):
            return metric(*metric_inputs)
        return metric(metric_inputs)

    def check_metrics(self, metrics: dict) -> None:
        for metric_name, metric in metrics.items():
            if not isinstance(metric, nn.Module):
                raise TypeError(
                    f"Metric {metric_name} must be an nn.Module, "
                    f"got {type(metric).__name__}."
                )

    def handle_error(self, message: str) -> None:
        if self.strict:
            raise RuntimeError(message)


class LoggableMetricSet(MetricSet):
    """
    MetricSet with flat log-dictionary formatting.
    """

    known_output_names: dict[str, tuple[str, ...]] = {}

    def to_log_dict(self, metrics_values: dict) -> dict:
        log_dict = {}

        for metric_name, metric_value in metrics_values.items():
            self._add_log_value(
                log_dict=log_dict,
                key=metric_name,
                metric_name=metric_name,
                value=metric_value,
            )

        return log_dict

    def _add_log_value(
        self,
        log_dict: dict,
        key: str,
        metric_name: str,
        value,
    ) -> None:
        if isinstance(value, dict):
            for sub_key, sub_value in value.items():
                self._add_log_value(
                    log_dict=log_dict,
                    key=f"{key}/{sub_key}",
                    metric_name=metric_name,
                    value=sub_value,
                )
            return

        if isinstance(value, (list, tuple)):
            for index, sub_value in enumerate(value):
                self._add_log_value(
                    log_dict=log_dict,
                    key=f"{key}/{index}",
                    metric_name=metric_name,
                    value=sub_value,
                )
            return

        if isinstance(value, torch.Tensor) and value.ndim != 0:
            output_names = self.known_output_names.get(metric_name)
            if output_names is None:
                self.handle_error(
                    f"Metric {metric_name} returned a non-scalar tensor with "
                    f"shape {tuple(value.shape)}, but no output names were provided."
                )
                return

            if len(output_names) != value.shape[-1]:
                self.handle_error(
                    f"Metric {metric_name} returned a tensor with last dimension "
                    f"{value.shape[-1]}, but {len(output_names)} output names "
                    "were provided."
                )
                return

            for index, output_name in enumerate(output_names):
                self._add_log_value(
                    log_dict=log_dict,
                    key=f"{key}/{output_name}",
                    metric_name=metric_name,
                    value=value[..., index],
                )
            return

        log_dict[key] = value


@METRIC_SET_REGISTRY.register_class(
    name="ac_video_jepa",
    default_config=DEFAULT_AC_VIDEO_JEPA_METRIC_SET_CONFIG,
    type_field="set_type",
    sub_builds=(METRICS_SUB_BUILD,),
)
class AcVideoJepaMetricSet(LoggableMetricSet):
    """
    MetricSet specialized for AcVideoJepa rollout outputs.
    """

    metric_to_input_names: dict[str, tuple[str, ...]] = {
        "prediction_loss": ("rollout_output",),
        "std_loss": ("rollout_output",),
        "cov_loss": ("rollout_output",),
        "sim_loss_t": ("rollout_output",),
        "idm_loss": ("rollout_output", "actions"),
    }

    valid_input_names = {"rollout_output", "actions"}

    def __init__(
        self,
        strict: bool = True,
        metric_to_input_names: dict[str, tuple[str, ...]] | None = None,
        metrics: dict[str, nn.Module] | None = None,
        **extra_metrics: nn.Module,
    ) -> None:
        super().__init__(
            strict=strict,
            metrics=metrics,
            **extra_metrics,
        )

        self.metric_to_input_names = dict(self.metric_to_input_names)

        if metric_to_input_names is not None:
            self.metric_to_input_names.update(metric_to_input_names)

        self.check_metric_routes()

    def forward(self, rollout_output, actions=None) -> dict:
        available_inputs = {
            "rollout_output": rollout_output,
            "actions": actions,
        }
        metrics_inputs = self.build_metrics_inputs(available_inputs)
        metrics_values = super().forward(metrics_inputs)
        return self.to_log_dict(metrics_values)

    def build_metrics_inputs(self, available_inputs: dict) -> dict:
        metrics_inputs = {}

        for metric_name in self.metrics.keys():
            input_names = self.get_metric_input_names(metric_name)
            metrics_inputs[metric_name] = {
                input_name: available_inputs[input_name]
                for input_name in input_names
            }

        return metrics_inputs

    def get_metric_input_names(self, metric_name: str) -> tuple[str, ...]:
        if metric_name not in self.metric_to_input_names:
            self.handle_error(f"No input route defined for metric {metric_name}.")
            return ()

        return self.metric_to_input_names[metric_name]

    def check_metric_routes(self) -> None:
        for metric_name in self.metrics.keys():
            input_names = self.get_metric_input_names(metric_name)
            for input_name in input_names:
                if input_name not in self.valid_input_names:
                    raise ValueError(
                        f"Invalid input name {input_name} for metric {metric_name}. "
                        f"Expected one of {self.valid_input_names}."
                    )
