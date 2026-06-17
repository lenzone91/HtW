import torch

from .metric_set import MetricSet
from .registry import METRIC_SET_REGISTRY, METRICS_SUB_BUILD


#############################################
# Loggable metric set
#############################################

DEFAULT_LOGGABLE_METRIC_SET_CONFIG = {
    "set_type": "loggable",
    "metrics": {},
}


@METRIC_SET_REGISTRY.register_class(
    name="loggable",
    default_config=DEFAULT_LOGGABLE_METRIC_SET_CONFIG,
    type_field="set_type",
    sub_builds=(METRICS_SUB_BUILD,),
)
class LoggableMetricSet(MetricSet):
    """
    Metric set with log-formatting utilities.

    Extends MetricSet with one responsibility: flatten raw metric outputs into a
    logging-ready dictionary. It does not add train/val/test prefixes and does
    not call Lightning logging directly.

    Vector-valued metric outputs require declared `known_output_names`; an
    undeclared or mismatched non-scalar output raises (strict).
    """

    known_output_names: dict[str, tuple[str, ...]] = {}

    def to_log_dict(self, metrics_values: dict) -> dict:
        """
        Convert raw metric outputs into a flat logging dictionary.

        Formatting:
            scalar      -> metric_name
            dict        -> metric_name/sub_key
            list/tuple  -> metric_name/output_name (if known) else metric_name/index
        """
        log_dict = {}

        for metric_name, metric_value in metrics_values.items():
            self._add_log_value(
                log_dict=log_dict,
                key=metric_name,
                metric_name=metric_name,
                value=metric_value,
            )

        return log_dict

    def _add_log_value(self, log_dict: dict, key: str, metric_name: str, value) -> None:
        if isinstance(value, dict):
            self._add_dict_output(log_dict, key, metric_name, value)
            return

        if isinstance(value, (list, tuple)):
            self._add_sequence_output(log_dict, key, metric_name, value)
            return

        if isinstance(value, torch.Tensor):
            self._add_tensor_output(log_dict, key, metric_name, value)
            return

        log_dict[key] = value

    def _add_tensor_output(
        self,
        log_dict: dict,
        key: str,
        metric_name: str,
        value: torch.Tensor,
    ) -> None:
        if value.ndim == 0:
            log_dict[key] = value
            return

        output_names = self.known_output_names.get(metric_name)

        if output_names is None:
            raise ValueError(
                f"Metric {metric_name} returned a non-scalar tensor with shape "
                f"{tuple(value.shape)}, but no output names were provided."
            )

        if len(output_names) != value.shape[-1]:
            raise ValueError(
                f"Metric {metric_name} returned a tensor with last dimension "
                f"{value.shape[-1]}, but {len(output_names)} output names were provided."
            )

        # Split vector-valued outputs along the last dimension.
        for index, output_name in enumerate(output_names):
            self._add_log_value(
                log_dict=log_dict,
                key=f"{key}/{output_name}",
                metric_name=metric_name,
                value=value[..., index],
            )

    def _add_dict_output(
        self,
        log_dict: dict,
        key: str,
        metric_name: str,
        value: dict,
    ) -> None:
        for sub_key, sub_value in value.items():
            self._add_log_value(
                log_dict=log_dict,
                key=f"{key}/{sub_key}",
                metric_name=metric_name,
                value=sub_value,
            )

    def _add_sequence_output(
        self,
        log_dict: dict,
        key: str,
        metric_name: str,
        value: list | tuple,
    ) -> None:
        output_names = self._get_output_names(metric_name, n_outputs=len(value))

        for index, sub_value in enumerate(value):
            sub_key = output_names[index] if output_names is not None else str(index)
            self._add_log_value(
                log_dict=log_dict,
                key=f"{key}/{sub_key}",
                metric_name=metric_name,
                value=sub_value,
            )

    def _get_output_names(
        self,
        metric_name: str,
        n_outputs: int,
    ) -> tuple[str, ...] | None:
        output_names = self.known_output_names.get(metric_name)

        if output_names is None:
            return None

        if len(output_names) != n_outputs:
            raise ValueError(
                f"Metric {metric_name} returned {n_outputs} values, "
                f"but {len(output_names)} output names were provided."
            )

        return output_names
