"""
Unit tests for AIML.Metrics.Metrics.base.BaseMetric.
"""

import pytest
import torch

from eb_jepa_cleaned.AIML.Metrics.Metrics.base import (
    BaseMetric,
)


class L1(BaseMetric):
    def forward(self, preds, target):
        self.check_same_shape(preds, target)
        return self.reduce((preds - target).abs())


def test_invalid_reduction_raises():
    with pytest.raises(ValueError, match="Invalid reduction"):
        L1(reduction="median")


def test_reduce_mean_sum_none():
    values = torch.tensor([1.0, 3.0])

    assert L1(reduction="mean").reduce(values) == 2.0
    assert L1(reduction="sum").reduce(values) == 4.0
    assert torch.equal(L1(reduction="none").reduce(values), values)


def test_forward_uses_reduction():
    metric = L1(reduction="mean")

    assert metric(torch.tensor([1.0, 2.0]), torch.tensor([0.0, 0.0])) == 1.5


def test_check_same_shape_raises():
    metric = L1()

    with pytest.raises(ValueError, match="same shape"):
        metric(torch.zeros(3), torch.zeros(4))


def test_check_not_autograd_tracked_raises_with_grad():
    metric = L1()
    x = torch.zeros(2, requires_grad=True)

    with pytest.raises(RuntimeError, match="evaluation-only"):
        metric.check_not_autograd_tracked(x)


def test_check_not_autograd_tracked_ok_without_grad():
    metric = L1()

    with torch.no_grad():
        x = torch.zeros(2, requires_grad=True)
        metric.check_not_autograd_tracked(x)  # no raise under no_grad
