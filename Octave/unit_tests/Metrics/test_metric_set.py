"""
Pytest test module for Metrics.MetricSet.

Pytest automatically discovers and runs functions whose names start with
``test_``. Therefore, no explicit test decorator is needed here.
"""

import pytest
import torch
from torch import nn

from Project_1_SepFormer_TSE_HF.idea_1_project_setup.src.Metrics.MetricSet import (
    LoggableMetricSet,
    MetricSet,
    TSEMetricSet,
)


#############################################
# Dummy metrics
#############################################

class IdentityMetric(nn.Module):
    """
    Return the input tensor unchanged.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


class SumMetric(nn.Module):
    """
    Return the sum of two tensors.
    """

    def forward(
        self,
        preds: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        return preds + target


class TSEInputCountMetric(nn.Module):
    """
    Return the number of provided TSE inputs.
    """

    def forward(
        self,
        preds: torch.Tensor,
        target: torch.Tensor,
        mixture: torch.Tensor,
    ) -> torch.Tensor:
        return torch.tensor(float(len([preds, target, mixture])))


#############################################
# Test data
#############################################

SCALAR = torch.tensor(1.0)
PRED = torch.tensor(1.0)
TARGET = torch.tensor(2.0)
MIXTURE = torch.tensor(3.0)


#############################################
# MetricSet validation tests
#############################################

def test_metric_set_rejects_non_module_metric() -> None:
    """
    Test that registered metrics must be nn.Module instances.
    """
    with pytest.raises(TypeError):
        MetricSet(
            strict=True,
            invalid_metric="not_a_module",
        )


def test_metric_set_strict_missing_metric_raises() -> None:
    """
    Test that strict mode rejects requested metrics that are not registered.
    """
    metric_set = MetricSet(
        strict=True,
        identity=IdentityMetric(),
    )

    with pytest.raises(RuntimeError):
        metric_set(
            metrics_inputs={
                "missing": SCALAR,
            },
        )


def test_metric_set_non_strict_missing_metric_is_skipped() -> None:
    """
    Test that non-strict mode skips requested metrics that are not registered.
    """
    metric_set = MetricSet(
        strict=False,
        identity=IdentityMetric(),
    )

    metric_values = metric_set(
        metrics_inputs={
            "identity": SCALAR,
            "missing": SCALAR,
        },
    )

    assert set(metric_values.keys()) == {"identity"}


#############################################
# MetricSet input dispatch tests
#############################################

def test_metric_set_calls_metric_with_tensor_input() -> None:
    """
    Test single-tensor metric input dispatch.
    """
    metric_set = MetricSet(
        strict=True,
        identity=IdentityMetric(),
    )

    metric_values = metric_set(
        metrics_inputs={
            "identity": SCALAR,
        },
    )

    assert torch.equal(metric_values["identity"], SCALAR)


def test_metric_set_calls_metric_with_tuple_input() -> None:
    """
    Test tuple metric input dispatch.
    """
    metric_set = MetricSet(
        strict=True,
        sum_metric=SumMetric(),
    )

    metric_values = metric_set(
        metrics_inputs={
            "sum_metric": (PRED, TARGET),
        },
    )

    assert torch.equal(metric_values["sum_metric"], PRED + TARGET)


def test_metric_set_calls_metric_with_list_input() -> None:
    """
    Test list metric input dispatch.
    """
    metric_set = MetricSet(
        strict=True,
        sum_metric=SumMetric(),
    )

    metric_values = metric_set(
        metrics_inputs={
            "sum_metric": [PRED, TARGET],
        },
    )

    assert torch.equal(metric_values["sum_metric"], PRED + TARGET)


def test_metric_set_calls_metric_with_dict_input() -> None:
    """
    Test keyword metric input dispatch.
    """
    metric_set = MetricSet(
        strict=True,
        sum_metric=SumMetric(),
    )

    metric_values = metric_set(
        metrics_inputs={
            "sum_metric": {
                "preds": PRED,
                "target": TARGET,
            },
        },
    )

    assert torch.equal(metric_values["sum_metric"], PRED + TARGET)


#############################################
# LoggableMetricSet tests
#############################################

def test_loggable_metric_set_flattens_dict_outputs() -> None:
    """
    Test that dictionary metric outputs are flattened.
    """
    metric_set = LoggableMetricSet(strict=True)

    log_dict = metric_set.to_log_dict(
        metrics_values={
            "metric": {
                "a": torch.tensor(1.0),
                "b": torch.tensor(2.0),
            },
        },
    )

    assert set(log_dict.keys()) == {"metric/a", "metric/b"}


def test_loggable_metric_set_flattens_sequence_outputs() -> None:
    """
    Test that sequence metric outputs are flattened using indices by default.
    """
    metric_set = LoggableMetricSet(strict=True)

    log_dict = metric_set.to_log_dict(
        metrics_values={
            "metric": (
                torch.tensor(1.0),
                torch.tensor(2.0),
            ),
        },
    )

    assert set(log_dict.keys()) == {"metric/0", "metric/1"}


def test_loggable_metric_set_rejects_unnamed_non_scalar_tensor() -> None:
    """
    Test that unnamed non-scalar tensor outputs fail in strict mode.
    """
    metric_set = LoggableMetricSet(strict=True)

    with pytest.raises(RuntimeError):
        metric_set.to_log_dict(
            metrics_values={
                "metric": torch.tensor([1.0, 2.0]),
            },
        )


def test_loggable_metric_set_flattens_named_non_scalar_tensor() -> None:
    """
    Test that named vector tensor outputs are flattened.
    """
    metric_set = LoggableMetricSet(strict=True)
    metric_set.known_output_names = {
        "metric": ("a", "b"),
    }

    log_dict = metric_set.to_log_dict(
        metrics_values={
            "metric": torch.tensor([1.0, 2.0]),
        },
    )

    assert set(log_dict.keys()) == {"metric/a", "metric/b"}


#############################################
# TSEMetricSet tests
#############################################

def test_tse_metric_set_builds_metric_inputs() -> None:
    """
    Test that TSEMetricSet builds metric inputs from TSE tensors.
    """
    metric_set = TSEMetricSet(
        strict=True,
        metric_to_input_names={
            "toy_metric": ("preds", "target", "mixture"),
        },
        toy_metric=TSEInputCountMetric(),
    )

    metric_values = metric_set(
        preds=PRED,
        target=TARGET,
        mixture=MIXTURE,
    )

    assert torch.equal(metric_values["toy_metric"], torch.tensor(3.0))


def test_tse_metric_set_strict_missing_required_input_raises() -> None:
    """
    Test that strict TSEMetricSet fails when a required input is missing.
    """
    metric_set = TSEMetricSet(
        strict=True,
        metric_to_input_names={
            "toy_metric": ("preds", "target", "mixture"),
        },
        toy_metric=TSEInputCountMetric(),
    )

    with pytest.raises(RuntimeError):
        metric_set(
            preds=PRED,
            target=TARGET,
            mixture=None,
        )


def test_tse_metric_set_rejects_invalid_input_route() -> None:
    """
    Test that invalid TSE input route names fail explicitly.
    """
    with pytest.raises(ValueError):
        TSEMetricSet(
            strict=True,
            metric_to_input_names={
                "toy_metric": ("invalid_input",),
            },
            toy_metric=IdentityMetric(),
        )