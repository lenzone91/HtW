"""
Unit tests for AIML.Metrics.MetricSets.metric_set.MetricSet and the factory.
"""

import pytest
import torch
from torch import nn

from src.AIML.Metrics.Metrics.base import (
    BaseMetric,
)
from src.AIML.Metrics.Metrics.registry import (
    METRIC_REGISTRY,
)
from src.AIML.Metrics.MetricSets.factory import (
    build_metric_set,
)
from src.AIML.Metrics.MetricSets.loggable_metric_set import (
    LoggableMetricSet,
)
from src.AIML.Metrics.MetricSets.metric_set import (
    MetricSet,
)


class SumMetric(nn.Module):
    def forward(self, x):
        return x.sum()


class DiffMetric(nn.Module):
    def forward(self, a, b):
        return (a - b).sum()


class KwMetric(nn.Module):
    def forward(self, preds, target):
        return (preds + target).sum()


def test_evaluate_dispatch_by_input_type():
    metric_set = MetricSet(metrics={"s": SumMetric(), "d": DiffMetric(), "k": KwMetric()})

    out = metric_set(
        {
            "s": torch.tensor([1.0, 2.0]),
            "d": (torch.tensor(5.0), torch.tensor(2.0)),
            "k": {"preds": torch.tensor(1.0), "target": torch.tensor(4.0)},
        }
    )

    assert out["s"] == 3.0
    assert out["d"] == 3.0
    assert out["k"] == 5.0


def test_unknown_metric_raises():
    metric_set = MetricSet(metrics={"s": SumMetric()})

    with pytest.raises(KeyError, match="not registered in this MetricSet"):
        metric_set({"missing": torch.tensor(1.0)})


def test_non_module_metric_raises():
    with pytest.raises(TypeError, match="must be an nn.Module"):
        MetricSet(metrics={"bad": "not a module"})


#############################################
# Factory + metrics sub-build
#############################################


class Constant(BaseMetric):
    def __init__(self, reduction="mean", value=1.0):
        super().__init__(reduction=reduction)
        self.value = value

    def forward(self, x):
        return torch.tensor(self.value)


@pytest.fixture
def registered_metric():
    name = "constant_metric_under_test"
    METRIC_REGISTRY.add_entry(
        name=name,
        object_cls=Constant,
        default_config={"reduction": "mean", "value": 1.0},
    )
    yield name
    METRIC_REGISTRY.entries.pop(name, None)


def test_build_metric_set_sub_builds_metrics(registered_metric):
    metric_set = build_metric_set(
        {"set_type": "metric", "metrics": {registered_metric: {"value": 7.0}}}
    )

    assert isinstance(metric_set, MetricSet)
    assert metric_set({registered_metric: torch.tensor(0.0)})[registered_metric] == 7.0


def test_build_metric_set_defaults_to_loggable(registered_metric):
    metric_set = build_metric_set({"metrics": {registered_metric: {}}})

    assert isinstance(metric_set, LoggableMetricSet)
