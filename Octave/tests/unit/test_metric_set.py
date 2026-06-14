import pytest
import torch
from torch import nn

from Octave.src.Metrics.metric_set import LoggableMetricSet, MetricSet


class IdentityMetric(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


class SumMetric(nn.Module):
    def forward(self, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        return left + right


def test_metric_set_rejects_non_module_metric() -> None:
    with pytest.raises(TypeError, match="must be an nn.Module"):
        MetricSet(invalid="not-a-module")


def test_metric_set_dispatches_dict_inputs() -> None:
    metric_set = MetricSet(sum_metric=SumMetric())

    values = metric_set(
        {
            "sum_metric": {
                "left": torch.tensor(1.0),
                "right": torch.tensor(2.0),
            }
        }
    )

    assert torch.equal(values["sum_metric"], torch.tensor(3.0))


def test_metric_set_rejects_missing_metric_in_strict_mode() -> None:
    metric_set = MetricSet(identity=IdentityMetric(), strict=True)

    with pytest.raises(RuntimeError, match="not registered"):
        metric_set({"missing": torch.tensor(1.0)})


def test_loggable_metric_set_flattens_dict_outputs() -> None:
    metric_set = LoggableMetricSet()

    log_dict = metric_set.to_log_dict(
        {
            "metric": {
                "left": torch.tensor(1.0),
                "right": torch.tensor(2.0),
            }
        }
    )

    assert set(log_dict) == {"metric/left", "metric/right"}
