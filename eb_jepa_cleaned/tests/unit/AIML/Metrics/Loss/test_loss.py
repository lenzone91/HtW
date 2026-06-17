"""
Unit tests for AIML.Metrics.Loss (WeightedMetricLoss + factory).
"""

import pytest
import torch

from eb_jepa_cleaned.AIML.Metrics.Loss.factory import (
    build_loss,
)
from eb_jepa_cleaned.AIML.Metrics.Loss.loss import (
    WeightedMetricLoss,
)


def test_weighted_sum_and_loss_terms():
    loss = WeightedMetricLoss(metric_weights={"a": -1.0, "b": 2.0})

    total, logs = loss({"a": torch.tensor(2.0), "b": torch.tensor(3.0)})

    assert total == pytest.approx(-2.0 + 6.0)
    assert logs["loss"] == total
    assert "loss/a" in logs and "loss/b" in logs


def test_inactive_weights_filtered():
    loss = WeightedMetricLoss(metric_weights={"a": 1.0, "zero": 0, "none": None})

    assert set(loss.metric_weights) == {"a"}


def test_missing_metric_raises():
    loss = WeightedMetricLoss(metric_weights={"a": 1.0})

    with pytest.raises(KeyError, match="required by the loss but is missing"):
        loss({"b": torch.tensor(1.0)})


def test_non_scalar_metric_value_raises():
    loss = WeightedMetricLoss(metric_weights={"a": 1.0})

    with pytest.raises(ValueError, match="must be scalar"):
        loss({"a": torch.tensor([1.0, 2.0])})


def test_non_real_weight_raises():
    with pytest.raises(TypeError, match="must be a real number"):
        WeightedMetricLoss(metric_weights={"a": "heavy"})


def test_return_loss_terms_false_only_returns_total():
    loss = WeightedMetricLoss(metric_weights={"a": 1.0}, return_loss_terms=False)

    _, logs = loss({"a": torch.tensor(1.0)})

    assert set(logs) == {"loss"}


def test_build_loss_factory():
    loss = build_loss(
        {"loss_type": "weighted_metric", "metric_weights": {"a": 1.0}}
    )

    assert isinstance(loss, WeightedMetricLoss)
    total, _ = loss({"a": torch.tensor(5.0)})
    assert total == 5.0
