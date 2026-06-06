"""
Pytest test module for Metrics.base.BaseMetric.

Pytest automatically discovers and runs functions whose names start with
``test_``. Therefore, no explicit test decorator is needed here.
"""

import pytest
import torch

from Project_1_SepFormer_TSE_HF.idea_1_project_setup.src.Metrics.base import (
    BaseMetric,
)


#############################################
# Test data
#############################################

VALUES = torch.tensor([1.0, 2.0, 3.0])


#############################################
# Reduction tests
#############################################

def test_base_metric_mean_reduction() -> None:
    """
    Test mean reduction.
    """
    metric = BaseMetric(reduction="mean")

    result = metric.reduce(VALUES)

    assert torch.isclose(result, torch.tensor(2.0))


def test_base_metric_sum_reduction() -> None:
    """
    Test sum reduction.
    """
    metric = BaseMetric(reduction="sum")

    result = metric.reduce(VALUES)

    assert torch.isclose(result, torch.tensor(6.0))


def test_base_metric_none_reduction() -> None:
    """
    Test identity reduction.
    """
    metric = BaseMetric(reduction="none")

    result = metric.reduce(VALUES)

    assert torch.equal(result, VALUES)


def test_base_metric_invalid_reduction_raises() -> None:
    """
    Test that invalid reductions fail explicitly.
    """
    with pytest.raises(ValueError):
        BaseMetric(reduction="invalid")


#############################################
# Shape validation tests
#############################################

def test_check_same_shape_accepts_matching_shapes() -> None:
    """
    Test that identical shapes pass same-shape validation.
    """
    metric = BaseMetric()

    metric.check_same_shape(
        preds=torch.randn(2, 1, 8),
        target=torch.randn(2, 1, 8),
    )


def test_check_same_shape_rejects_mismatching_shapes() -> None:
    """
    Test that different shapes fail same-shape validation.
    """
    metric = BaseMetric()

    with pytest.raises(ValueError):
        metric.check_same_shape(
            preds=torch.randn(2, 1, 8),
            target=torch.randn(2, 1, 9),
        )


def test_check_single_channel_accepts_single_waveform() -> None:
    """
    Test that shape (T,) is accepted as single-channel audio.
    """
    metric = BaseMetric()

    metric.check_single_channel(torch.randn(8))


def test_check_single_channel_accepts_batched_waveform() -> None:
    """
    Test that shape (B, T) is accepted as single-channel audio.
    """
    metric = BaseMetric()

    metric.check_single_channel(torch.randn(2, 8))


def test_check_single_channel_accepts_explicit_single_channel_batch() -> None:
    """
    Test that shape (B, 1, T) is accepted as single-channel audio.
    """
    metric = BaseMetric()

    metric.check_single_channel(torch.randn(2, 1, 8))


def test_check_single_channel_rejects_multi_channel_batch() -> None:
    """
    Test that shape (B, C, T) with C != 1 is rejected.
    """
    metric = BaseMetric()

    with pytest.raises(ValueError):
        metric.check_single_channel(torch.randn(2, 2, 8))


#############################################
# Autograd validation tests
#############################################

def test_check_not_autograd_tracked_accepts_non_grad_tensor() -> None:
    """
    Test that tensors without gradient tracking are accepted.
    """
    metric = BaseMetric()
    x = torch.randn(2, 8)

    metric.check_not_autograd_tracked(x)


def test_check_not_autograd_tracked_rejects_grad_tensor() -> None:
    """
    Test that tensors requiring gradients fail when grad mode is enabled.
    """
    metric = BaseMetric()
    x = torch.randn(2, 8, requires_grad=True)

    with pytest.raises(RuntimeError):
        metric.check_not_autograd_tracked(x)


def test_check_not_autograd_tracked_accepts_grad_tensor_under_no_grad() -> None:
    """
    Test that gradient-requiring tensors are accepted under torch.no_grad().
    """
    metric = BaseMetric()
    x = torch.randn(2, 8, requires_grad=True)

    with torch.no_grad():
        metric.check_not_autograd_tracked(x)


#############################################
# Helper tests
#############################################

def test_to_single_channel_shape_removes_channel_dimension() -> None:
    """
    Test that (B, 1, T) is converted to (B, T).
    """
    metric = BaseMetric()
    x = torch.randn(2, 1, 8)

    y = metric._to_single_channel_shape(x)

    assert y.shape == torch.Size([2, 8])


def test_to_single_channel_shape_keeps_flat_shape() -> None:
    """
    Test that already flat single-channel inputs are unchanged.
    """
    metric = BaseMetric()
    x = torch.randn(2, 8)

    y = metric._to_single_channel_shape(x)

    assert y.shape == x.shape


def test_to_flat_batch_returns_flattened_tensor_and_batch_shape() -> None:
    """
    Test that leading dimensions are flattened and returned separately.
    """
    metric = BaseMetric()
    x = torch.randn(2, 3, 8)

    y, batch_shape = metric._to_flat_batch(x)

    assert y.shape == torch.Size([6, 8])
    assert batch_shape == torch.Size([2, 3])