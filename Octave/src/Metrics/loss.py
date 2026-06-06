import numbers

import torch
from torch import nn

from ..Utils.error import handle_error


class WeightedMetricLoss(nn.Module):
    """
    Weighted sum over a flat metric dictionary.

    Convention:
        The returned loss is minimized.

        - positive weight -> metric should be minimized;
        - negative weight -> metric should be maximized.

        Metrics with weight 0 or None are considered inactive and are removed
        from metric_weights at construction time.
    """

    def __init__(
        self,
        metric_weights: dict[str, float | None],
        strict: bool = True,
        name: str = "loss",
        return_loss_terms: bool = True,
        
    ):
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
            return {
                self.name: loss,
                **loss_terms,
            }

        return {
            self.name: loss,
        }

    def filter_metric_weights(
        self,
        metric_weights: dict[str, float | None],
    ) -> dict[str, float]:
        """
        Remove inactive loss terms.

        Convention:
            weight is None -> inactive metric
            weight == 0   -> inactive metric
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
        if len(loss_terms) == 0:
            raise ValueError("No valid metric was available to compute the loss.")

        return sum(loss_terms.values())

    def check_metric_weights(self) -> None:
        for metric_name, weight in self.metric_weights.items():
            self.check_metric_name(metric_name)
            self.check_metric_weight(metric_name, weight)

    def check_metric_name(self, metric_name: str) -> None:
        if not isinstance(metric_name, str):
            raise TypeError(
                f"Metric names in metric_weights must be strings, got {type(metric_name)}."
            )

    def check_metric_weight(
        self,
        metric_name: str,
        weight: float,
    ) -> None:
        if not isinstance(weight, numbers.Real):
            raise TypeError(
                f"Weight for metric {metric_name} must be a real number, got {type(weight)}."
            )

    def check_metric_value(
        self,
        metric_name: str,
        metric_value: torch.Tensor,
    ) -> None:
        if not isinstance(metric_value, torch.Tensor):
            raise TypeError(
                f"Metric value {metric_name} must be a torch.Tensor, got {type(metric_value)}."
            )

        if metric_value.ndim != 0:
            raise ValueError(
                f"Metric value {metric_name} must be scalar, got shape {metric_value.shape}."
            )

    def handle_error(self, msg: str) -> None:
        handle_error(msg, self.strict)


    def __str__(self) -> str:
        # Get the maximum name_length to format the message so that the "|" and "=" are vertically aligned
        max_name_length = max(
            (len(metric_name) for metric_name in self.metric_weights),
            default=0,
        )

        mes = "Loss with:\n"

        for metric_name, metric_weight in self.metric_weights.items():
            mes += f"{metric_name:<{max_name_length}} | w = {metric_weight}\n"

        return mes