import numbers

import torch
from torch import nn

from .configs import DEFAULT_WEIGHTED_METRIC_LOSS_CONFIG
from .registry import LOSS_REGISTRY

@LOSS_REGISTRY.register_class(
    name="weighted_metric",
    default_config=DEFAULT_WEIGHTED_METRIC_LOSS_CONFIG,
    type_field="loss_type",
)
class WeightedMetricLoss(nn.Module):
    """
    Weighted sum over a flat metric dictionary.
    """

    def __init__(
        self,
        metric_weights: dict[str, float | None],
        strict: bool = True,
        name: str = "loss",
        return_loss_terms: bool = True,
    ) -> None:
        super().__init__()
        self.metric_weights = self.filter_metric_weights(metric_weights)
        self.strict = strict
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
            if metric_value is None:
                continue

            self.check_metric_value(metric_name, metric_value)
            loss_terms[f"{self.name}/{metric_name}"] = weight * metric_value

        loss = self.sum_loss_terms(loss_terms)
        return loss, self.build_loss_logs(loss, loss_terms)

    def build_loss_logs(
        self,
        loss: torch.Tensor,
        loss_terms: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        if self.return_loss_terms:
            return {
                self.name: loss,
                **loss_terms,
            }

        return {self.name: loss}

    def filter_metric_weights(
        self,
        metric_weights: dict[str, float | None],
    ) -> dict[str, float]:
        return {
            metric_name: weight
            for metric_name, weight in metric_weights.items()
            if weight is not None and weight != 0
        }

    def get_metric_value(
        self,
        metric_name: str,
        metric_values: dict[str, torch.Tensor],
    ) -> torch.Tensor | None:
        if metric_name in metric_values:
            return metric_values[metric_name]

        self.handle_error(
            f"Metric {metric_name} is required by the loss but is missing."
        )
        return None

    def sum_loss_terms(
        self,
        loss_terms: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        if not loss_terms:
            raise ValueError("No valid metric was available to compute the loss.")

        return sum(loss_terms.values())

    def check_metric_weights(self) -> None:
        for metric_name, weight in self.metric_weights.items():
            if not isinstance(metric_name, str):
                raise TypeError(
                    "Metric names in metric_weights must be strings, "
                    f"got {type(metric_name).__name__}."
                )
            if not isinstance(weight, numbers.Real):
                raise TypeError(
                    f"Weight for metric {metric_name} must be a real number, "
                    f"got {type(weight).__name__}."
                )

    def check_metric_value(
        self,
        metric_name: str,
        metric_value: torch.Tensor,
    ) -> None:
        if not isinstance(metric_value, torch.Tensor):
            raise TypeError(
                f"Metric value {metric_name} must be a torch.Tensor, "
                f"got {type(metric_value).__name__}."
            )

        if metric_value.ndim != 0:
            raise ValueError(
                f"Metric value {metric_name} must be scalar, "
                f"got shape {metric_value.shape}."
            )

    def handle_error(self, message: str) -> None:
        if self.strict:
            raise RuntimeError(message)
