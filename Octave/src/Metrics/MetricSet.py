import torch
from torch import nn

from ..Utils.error import handle_error

#############################################
# MetricSet
#############################################

# Just a collection of metrics with appropriate testing and handling
# This is meant to be specilized but was split to keep the code structure 
# readable


class MetricSet(nn.Module):
    """
    Collection of metric modules.

    General-purpose metric orchestration:
        - stores named metrics;
        - checks that registered metrics are valid nn.Module objects;
        - evaluates each metric on its own inputs;
        - handles missing metrics in strict or non-strict mode.

    This class does not know about Lightning logging.
    This class does not know about train / validation / test.
    """

    def __init__(
        self,
        strict: bool = True,
        **metrics: nn.Module,
    ):
        super().__init__()

        self.strict = strict
        self.check_metrics(metrics)

        # ModuleDict ensures that metric modules are properly registered.
        self.metrics = nn.ModuleDict(metrics)

    def forward(self, metrics_inputs: dict) -> dict:
        """
        Evaluate all requested metrics.

        Expected format:
            metrics_inputs[metric_name] = metric input specification

        Input dispatch convention:
            tensor       -> metric(x)
            tuple/list   -> metric(*inputs)
            dict         -> metric(**inputs)
        """
        return self.evaluate_metrics(metrics_inputs)

    def evaluate_metrics(self, metrics_inputs: dict) -> dict:
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
        return self.call_metric(metric, metric_inputs)

    def call_metric(self, metric: nn.Module, metric_inputs):
        # Keyword-input metrics, e.g. metric(preds=..., target=...).
        if isinstance(metric_inputs, dict):
            return metric(**metric_inputs)

        # Multi-input metrics, e.g. metric(preds, target, mixture).
        if isinstance(metric_inputs, (tuple, list)):
            return metric(*metric_inputs)

        # Single-input metrics, e.g. metric(x).
        return metric(metric_inputs)

    def check_metrics(self, metrics: dict) -> None:
        for metric_name, metric in metrics.items():
            if not isinstance(metric, nn.Module):
                raise TypeError(
                    f"Metric {metric_name} must be an nn.Module, "
                    f"got {type(metric)}."
                )

    def handle_error(self, msg: str) -> None:
        handle_error(msg, self.strict)

    def __str__(self) -> str:
        metric_names = "\n".join(self.metrics.keys())
        return f"MetricSet with:\n{metric_names}"

    
#######################################################
# LoggableMetricSet
#######################################################

# Some of the metrics do not return scalar values (cf DNSMOS P835)
# Therefore, if we want to register the metrics values properly we should reshape
# the raw metric evaluation dictionnary. 

class LoggableMetricSet(MetricSet):
    """
    Metric set with log-formatting utilities.

    Extends MetricSet with one extra responsibility:
        convert raw metric outputs into a flat dictionary suitable for logging.

    It does not add train / validation / test prefixes.
    It does not call PyTorch Lightning logging methods directly.
    """

    known_output_names: dict[str, tuple[str, ...]] = {}

    def to_log_dict(self, metrics_values: dict) -> dict:
        """
        Convert raw metric outputs into a flat logging dictionary.

        Formatting convention:
            scalar output      -> metric_name
            dict output        -> metric_name/sub_key
            list/tuple output  -> metric_name/output_name if known,
                                  otherwise metric_name/index
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

    def _add_log_value(
        self,
        log_dict: dict,
        key: str,
        metric_name: str,
        value,
    ) -> None:
        # Dictionary outputs already provide meaningful submetric names.
        if isinstance(value, dict):
            self._add_dict_output(log_dict, key, metric_name, value)
            return

        # Sequence outputs are expanded using known names or fallback indices.
        if isinstance(value, (list, tuple)):
            self._add_sequence_output(log_dict, key, metric_name, value)
            return

        # Tensor outputs may still be non-scalar, e.g. DNSMOS P835 -> (..., 3).
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
        # Scalar tensors are directly compatible with Lightning logging.
        if value.ndim == 0:
            log_dict[key] = value
            return

        output_names = self.known_output_names.get(metric_name)

        if output_names is None:
            self.handle_error(
                f"Metric {metric_name} returned a non-scalar tensor with shape "
                f"{tuple(value.shape)}, but no output names were provided."
            )
            return

        if len(output_names) != value.shape[-1]:
            self.handle_error(
                f"Metric {metric_name} returned a tensor with last dimension "
                f"{value.shape[-1]}, but {len(output_names)} output names were provided."
            )
            return

        # Split vector-valued metric outputs along the last dimension.
        # The original non-scalar tensor is not added to log_dict.
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
        output_names = self._get_output_names(
            metric_name=metric_name,
            n_outputs=len(value),
        )

        for index, sub_value in enumerate(value):
            sub_key = (
                output_names[index]
                if output_names is not None
                else str(index)
            )

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
            self.handle_error(
                f"Metric {metric_name} returned {n_outputs} values, "
                f"but {len(output_names)} output names were provided."
            )
            return None

        return output_names
    


##########################################
# TSEMetricSet
##########################################

# While the LightningMetricSet could be used in itself, it requires a heavy input
# where each link metric to input has to be specified. 
# In TSE we only have  ?? 4 ?? object of interests : preds, target, mixture, clues
# Additionnally, the metric that we will use are known
# We exploit that to make an object that does the heavy metric to input links for the user

# /!\ Remember to update the maps when new metrics are added /!\

class TSEMetricSet(LoggableMetricSet):
    """
    Metric set specialized for Target Speech Extraction.

    TSE convention:
        preds   = estimated/extracted speech
        target  = clean reference target speech
        mixture = input mixture
        clue    = optional target clue / enrollment information

    This class builds metric inputs automatically from TSE tensors.
    """

    valid_input_names = {"preds", "target", "mixture", "clue"}

    metric_to_input_names: dict[str, tuple[str, ...]] = {
        # Intrusive waveform metrics.
        "snr": ("preds", "target"),
        "sisdr": ("preds", "target"),
        "sdsdr": ("preds", "target"),
        "dtw": ("preds", "target"),
        "lp_error": ("preds", "target"),

        # Scale-invariant decomposition metrics.
        "sisir": ("preds", "target", "mixture"),
        "sisar": ("preds", "target", "mixture"),

        # Spectral intrusive metrics.
        "lsd": ("preds", "target"),
        "spectral_kl": ("preds", "target"),
        "itakura_saito": ("preds", "target"),

        # Perceptual intrusive metrics.
        "stoi": ("preds", "target"),
        "estoi": ("preds", "target"),
        "pesq": ("preds", "target"),

        # Non-intrusive perceptual metrics.
        "dnsmos": ("preds",),
        "dnsmos_p808": ("preds",),
        "dnsmos_p835": ("preds",),
        "dnsmos_sig": ("preds",),
        "dnsmos_bak": ("preds",),
        "dnsmos_ovrl": ("preds",),
    }

    known_output_names: dict[str, tuple[str, ...]] = {
        # Only useful if these metrics return several explicit values.
        "dnsmos": ("p808", "sig", "bak", "ovrl"),
        "dnsmos_p835": ("sig", "bak", "ovrl"),
    }

    def __init__(
        self,
        strict: bool = True,
        metric_to_input_names: dict[str, tuple[str, ...]] | None = None,
        **metrics: nn.Module,
    ):
        super().__init__(
            strict=strict,
            **metrics,
        )

        self.metric_to_input_names = dict(self.metric_to_input_names)

        # User-provided routes override default TSE routes.
        if metric_to_input_names is not None:
            self.metric_to_input_names.update(metric_to_input_names)

        self.check_metric_routes()

    def forward(
        self,
        preds: torch.Tensor,
        target: torch.Tensor | None = None,
        mixture: torch.Tensor | None = None,
        clue=None,
    ) -> dict:
        """
        Evaluate registered TSE metrics and return a Lightning-loggable dict.
        """
        available_inputs = {
            "preds": preds,
            "target": target,
            "mixture": mixture,
            "clue": clue,
        }

        metrics_inputs = self.build_metrics_inputs(available_inputs)
        metrics_values = super().forward(metrics_inputs)
        return self.to_log_dict(metrics_values)

    def build_metrics_inputs(self, available_inputs: dict) -> dict:
        """
        Build the explicit MetricSet input dictionary from TSE tensors.
        """
        metrics_inputs = {}

        for metric_name in self.metrics.keys():
            input_names = self.get_metric_input_names(metric_name)

            metrics_inputs[metric_name] = {
                input_name: self.get_available_input(
                    available_inputs=available_inputs,
                    metric_name=metric_name,
                    input_name=input_name,
                )
                for input_name in input_names
            }

        return metrics_inputs

    def get_metric_input_names(self, metric_name: str) -> tuple[str, ...]:
        if metric_name not in self.metric_to_input_names:
            self.handle_error(
                f"No input route defined for metric {metric_name}."
            )
            return ()

        return self.metric_to_input_names[metric_name]

    def get_available_input(
        self,
        available_inputs: dict,
        metric_name: str,
        input_name: str,
    ):
        value = available_inputs[input_name]

        if value is None:
            self.handle_error(
                f"Metric {metric_name} requires input {input_name}, "
                "but it was not provided."
            )

        return value

    def check_metric_routes(self) -> None:
        """
        Check that every registered metric has a valid input route.
        """
        for metric_name in self.metrics.keys():
            input_names = self.get_metric_input_names(metric_name)
            self.check_input_names(metric_name, input_names)

    def check_input_names(
        self,
        metric_name: str,
        input_names: tuple[str, ...],
    ) -> None:
        for input_name in input_names:
            if input_name not in self.valid_input_names:
                raise ValueError(
                    f"Invalid input name {input_name} for metric {metric_name}. "
                    f"Expected one of {self.valid_input_names}."
                )