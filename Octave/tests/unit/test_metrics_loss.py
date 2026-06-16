import pytest
import torch

from Octave.src.Metrics.Loss.loss import WeightedMetricLoss


def test_weighted_metric_loss_filters_inactive_weights() -> None:
    loss = WeightedMetricLoss(
        metric_weights={
            "active": 1.0,
            "none_inactive": None,
            "zero_inactive": 0.0,
        }
    )

    assert loss.metric_weights == {"active": 1.0}


def test_weighted_metric_loss_sums_weighted_metric_values() -> None:
    loss_object = WeightedMetricLoss(
        metric_weights={
            "metric_to_minimize": 2.0,
            "metric_to_maximize": -4.0,
        },
        name="loss",
    )

    loss, loss_logs = loss_object(
        {
            "metric_to_minimize": torch.tensor(2.0),
            "metric_to_maximize": torch.tensor(3.0),
        }
    )

    assert torch.isclose(loss, torch.tensor(-8.0))
    assert torch.isclose(loss_logs["loss"], torch.tensor(-8.0))
    assert torch.isclose(loss_logs["loss/metric_to_minimize"], torch.tensor(4.0))
    assert torch.isclose(loss_logs["loss/metric_to_maximize"], torch.tensor(-12.0))


def test_weighted_metric_loss_rejects_missing_metric() -> None:
    loss_object = WeightedMetricLoss(metric_weights={"missing": 1.0})

    with pytest.raises(RuntimeError, match="required by the loss"):
        loss_object({"available": torch.tensor(1.0)})


def test_weighted_metric_loss_rejects_non_scalar_metric_value() -> None:
    loss_object = WeightedMetricLoss(metric_weights={"metric": 1.0})

    with pytest.raises(ValueError, match="must be scalar"):
        loss_object({"metric": torch.tensor([1.0, 2.0])})
