import numbers

import torch
from torch import nn

from .registry import LOSS_REGISTRY


#############################################
# Weighted metric loss
#############################################

# default_config is the registry key allow-list; metric_weights inner keys are
# user-defined (not validated by the allow-list, which checks top-level keys).

DEFAULT_WEIGHTED_METRIC_LOSS_CONFIG = {
    "loss_type": "weighted_metric",
    "name": "loss",
    "return_loss_terms": True,
    "metric_weights": {},
}


@LOSS_REGISTRY.register_class(
    name="weighted_metric",
    default_config=DEFAULT_WEIGHTED_METRIC_LOSS_CONFIG,
    type_field="loss_type",
)
class WeightedMetricLoss(nn.Module):
    """
    Weighted sum over a flat metric dictionary.

    Convention (the returned loss is minimized):
        - positive weight -> metric should be minimized;
        - negative weight -> metric should be maximized;
        - weight 0 or None -> inactive metric, removed at construction.

    Strict: a metric required by the weights but missing at forward time raises.
    """

    def __init__(
        self,
        metric_weights: dict[str, float | None],
        name: str = "loss",
        return_loss_terms: bool = True,
    ):
        super().__init__()

        self.metric_weights = self.filter_metric_weights(metric_weights)
        self.name = name
        self.return_loss_terms = return_loss_terms

        self.check_metric_weights()

    def forward(
        self,
        metric_values: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        loss_terms = {}

        for metric_name, weight in self.metric_weights.items():
            metric_value = self.get_metric_value(metric_name, metric_values)
            self.check_metric_value(metric_name, metric_value)

            # Log-ready loss component.
            loss_terms[f"{self.name}/{metric_name}"] = weight * metric_value

        loss = self.sum_loss_terms(loss_terms)
        loss_logs = self.build_loss_logs(loss, loss_terms)

        return loss, loss_logs

    def build_loss_logs(
        self,
        loss: torch.Tensor,
        loss_terms: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        if self.return_loss_terms:
            return {self.name: loss, **loss_terms}

        return {self.name: loss}

    def filter_metric_weights(
        self,
        metric_weights: dict[str, float | None],
    ) -> dict[str, float]:
        """
        Remove inactive loss terms (weight None or 0).
        """
        return {
            metric_name: weight
            for metric_name, weight in metric_weights.items()
            if weight is not None and weight != 0
        }

    def get_metric_value(
        self,
        metric_name: str,
        metric_values: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        if metric_name not in metric_values:
            raise KeyError(
                f"Metric '{metric_name}' is required by the loss but is missing."
            )

        return metric_values[metric_name]

    def sum_loss_terms(
        self,
        loss_terms: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        if len(loss_terms) == 0:
            raise ValueError("No valid metric was available to compute the loss.")

        return sum(loss_terms.values())

    #############################################
    # Validation helpers
    #############################################

    def check_metric_weights(self) -> None:
        for metric_name, weight in self.metric_weights.items():
            self.check_metric_name(metric_name)
            self.check_metric_weight(metric_name, weight)

    def check_metric_name(self, metric_name: str) -> None:
        if not isinstance(metric_name, str):
            raise TypeError(
                "Metric names in metric_weights must be strings, "
                f"got {type(metric_name)}."
            )

    def check_metric_weight(self, metric_name: str, weight: float) -> None:
        if not isinstance(weight, numbers.Real):
            raise TypeError(
                f"Weight for metric {metric_name} must be a real number, "
                f"got {type(weight)}."
            )

    def check_metric_value(
        self,
        metric_name: str,
        metric_value: torch.Tensor,
    ) -> None:
        if not isinstance(metric_value, torch.Tensor):
            raise TypeError(
                f"Metric value {metric_name} must be a torch.Tensor, "
                f"got {type(metric_value)}."
            )

        if metric_value.ndim != 0:
            raise ValueError(
                f"Metric value {metric_name} must be scalar, "
                f"got shape {metric_value.shape}."
            )

    def __str__(self) -> str:
        max_name_length = max(
            (len(metric_name) for metric_name in self.metric_weights),
            default=0,
        )

        message = "Loss with:\n"
        for metric_name, metric_weight in self.metric_weights.items():
            message += f"{metric_name:<{max_name_length}} | w = {metric_weight}\n"

        return message
