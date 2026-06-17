"""
Unit tests for AIML.Metrics.Metrics.factory.
"""

import pytest
import torch

from src.AIML.Metrics.Metrics.base import (
    BaseMetric,
)
from src.AIML.Metrics.Metrics.factory import (
    build_metric,
    build_metrics,
)
from src.AIML.Metrics.Metrics.registry import (
    METRIC_REGISTRY,
)
from src.Workflow.Factory.errors import (
    RegistryError,
)


class Scaled(BaseMetric):
    def __init__(self, reduction="mean", scale=1.0):
        super().__init__(reduction=reduction)
        self.scale = scale

    def forward(self, preds, target):
        return self.reduce((preds - target).abs()) * self.scale


@pytest.fixture
def registered_metric():
    name = "scaled_metric_under_test"
    METRIC_REGISTRY.add_entry(
        name=name,
        object_cls=Scaled,
        default_config={"reduction": "mean", "scale": 1.0},
    )
    yield name
    METRIC_REGISTRY.entries.pop(name, None)


def test_build_metric_by_name(registered_metric):
    metric = build_metric({"scale": 2.0}, metric_name=registered_metric)

    assert isinstance(metric, Scaled)
    assert metric(torch.tensor([1.0]), torch.tensor([0.0])) == 2.0


def test_build_metrics_named(registered_metric):
    metrics = build_metrics({registered_metric: {"scale": 1.0}})

    assert set(metrics) == {registered_metric}
    assert isinstance(metrics[registered_metric], Scaled)


def test_build_metric_unknown_name_raises():
    with pytest.raises(RegistryError, match="Unknown metric"):
        build_metric({}, metric_name="missing")
