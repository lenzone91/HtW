"""
Pytest test module for Metrics.loss.WeightedMetricLoss.

Pytest automatically discovers and runs functions whose names start with
``test_``. Therefore, no explicit test decorator is needed here.
"""

import pytest
import torch

from Project_1_SepFormer_TSE_HF.idea_1_project_setup.src.Metrics.loss import (
    WeightedMetricLoss,
)


#############################################
# Test data
#############################################

METRIC_VALUES = {
    "metric_to_minimize": torch.tensor(2.0),
    "metric_to_maximize": torch.tensor(3.0),
}


#############################################
# Weight handling tests
#############################################

def test_loss_filters_inactive_weights() -> None:
    """
    Test that None and zero weights are removed at construction time.
    """
    loss_object = WeightedMetricLoss(
        metric_weights={
            "active": 1.0,
            "none_inactive": None,
            "zero_inactive": 0.0,
        },
    )

    assert loss_object.metric_weights == {"active": 1.0}


def test_loss_positive_and_negative_weights() -> None:
    """
    Test the minimization/maximization sign convention.
    """
    loss_object = WeightedMetricLoss(
        metric_weights={
            "metric_to_minimize": 2.0,
            "metric_to_maximize": -4.0,
        },
        name="loss",
    )

    loss, loss_logs = loss_object(METRIC_VALUES)

    expected_loss = torch.tensor(-8.0)

    assert torch.isclose(loss, expected_loss)
    assert torch.isclose(loss_logs["loss"], expected_loss)
    assert torch.isclose(loss_logs["loss/metric_to_minimize"], torch.tensor(4.0))
    assert torch.isclose(loss_logs["loss/metric_to_maximize"], torch.tensor(-12.0))


#############################################
# Logging tests
#############################################

def test_loss_returns_loss_terms_by_default() -> None:
    """
    Test that total loss and weighted loss terms are returned by default.
    """
    loss_object = WeightedMetricLoss(
        metric_weights={
            "metric_to_minimize": 1.0,
        },
        name="loss",
        return_loss_terms=True,
    )

    _, loss_logs = loss_object(METRIC_VALUES)

    assert set(loss_logs.keys()) == {
        "loss",
        "loss/metric_to_minimize",
    }


def test_loss_can_return_only_total_loss_log() -> None:
    """
    Test that loss terms can be hidden from the returned log dictionary.
    """
    loss_object = WeightedMetricLoss(
        metric_weights={
            "metric_to_minimize": 1.0,
        },
        name="loss",
        return_loss_terms=False,
    )

    _, loss_logs = loss_object(METRIC_VALUES)

    assert set(loss_logs.keys()) == {"loss"}


#############################################
# Missing metric tests
#############################################

def test_loss_strict_missing_metric_raises() -> None:
    """
    Test that strict mode rejects missing required metric values.
    """
    loss_object = WeightedMetricLoss(
        metric_weights={
            "missing_metric": 1.0,
        },
        strict=True,
    )

    with pytest.raises(RuntimeError):
        loss_object(METRIC_VALUES)


def test_loss_non_strict_missing_metric_is_skipped() -> None:
    """
    Test that non-strict mode skips missing metric values.
    """
    loss_object = WeightedMetricLoss(
        metric_weights={
            "metric_to_minimize": 1.0,
            "missing_metric": 1.0,
        },
        strict=False,
    )

    loss, loss_logs = loss_object(METRIC_VALUES)

    assert torch.isclose(loss, torch.tensor(2.0))
    assert set(loss_logs.keys()) == {
        "loss",
        "loss/metric_to_minimize",
    }


def test_loss_raises_when_no_valid_metric_is_available() -> None:
    """
    Test that loss computation fails when no active metric value is available.
    """
    loss_object = WeightedMetricLoss(
        metric_weights={
            "missing_metric": 1.0,
        },
        strict=False,
    )

    with pytest.raises(ValueError):
        loss_object(METRIC_VALUES)


#############################################
# Input validation tests
#############################################

def test_loss_rejects_non_string_metric_name() -> None:
    """
    Test that metric weight keys must be strings.
    """
    with pytest.raises(TypeError):
        WeightedMetricLoss(
            metric_weights={
                1: 1.0,
            },
        )


def test_loss_rejects_non_real_weight() -> None:
    """
    Test that active metric weights must be real numbers.
    """
    with pytest.raises(TypeError):
        WeightedMetricLoss(
            metric_weights={
                "metric": "invalid",
            },
        )


def test_loss_rejects_non_tensor_metric_value() -> None:
    """
    Test that metric values must be torch tensors.
    """
    loss_object = WeightedMetricLoss(
        metric_weights={
            "metric": 1.0,
        },
    )

    with pytest.raises(TypeError):
        loss_object(
            metric_values={
                "metric": 1.0,
            },
        )


def test_loss_rejects_non_scalar_metric_value() -> None:
    """
    Test that metric values must be scalar tensors.
    """
    loss_object = WeightedMetricLoss(
        metric_weights={
            "metric": 1.0,
        },
    )

    with pytest.raises(ValueError):
        loss_object(
            metric_values={
                "metric": torch.tensor([1.0, 2.0]),
            },
        )