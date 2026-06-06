"""
Pytest test module for Metrics.subjective.

Pytest automatically discovers and runs functions whose names start with
``test_``. Therefore, no explicit test decorator is needed here.

Subjective metric dependencies are not skipped: if a required backend is
missing, the test should fail explicitly.
"""

import pytest
import torch

from Project_1_SepFormer_TSE_HF.idea_1_project_setup.src.Metrics.subjective import (
    DNSMOSBAK,
    DNSMOSOVRL,
    DNSMOSP808,
    DNSMOSP835,
    DNSMOSSIG,
    ESTOI,
    PESQ,
    STOI,
)


#############################################
# Test data
#############################################

BATCH_SIZE = 2
CHANNELS = 1
SIGNAL_LENGTH = 16000
SAMPLE_RATE = 16000

PREDS = torch.randn(BATCH_SIZE, CHANNELS, SIGNAL_LENGTH)
TARGET = torch.randn(BATCH_SIZE, CHANNELS, SIGNAL_LENGTH)


#############################################
# Shared checks
#############################################

def check_metric_returns_scalar(metric, *args) -> None:
    """
    Check that a subjective metric returns a scalar tensor under mean reduction.
    """
    with torch.no_grad():
        value = metric(*args)

    assert isinstance(value, torch.Tensor)
    assert value.ndim == 0


def check_metric_supports_none_reduction(metric, *args) -> None:
    """
    Check that a subjective metric returns one value per batch item under none reduction.
    """
    with torch.no_grad():
        value = metric(*args)

    assert isinstance(value, torch.Tensor)
    assert value.shape == torch.Size([BATCH_SIZE])


#############################################
# STOI / ESTOI tests
#############################################

def test_stoi_forward_returns_scalar() -> None:
    """
    Test STOI forward pass under mean reduction.
    """
    metric = STOI(
        sample_rate=SAMPLE_RATE,
        reduction="mean",
    )

    check_metric_returns_scalar(metric, PREDS, TARGET)


def test_estoi_forward_returns_scalar() -> None:
    """
    Test ESTOI forward pass under mean reduction.
    """
    metric = ESTOI(
        sample_rate=SAMPLE_RATE,
        reduction="mean",
    )

    check_metric_returns_scalar(metric, PREDS, TARGET)


def test_stoi_rejects_autograd_tracked_inputs() -> None:
    """
    Test that STOI rejects tensors requiring gradients in grad mode.
    """
    metric = STOI(
        sample_rate=SAMPLE_RATE,
        reduction="mean",
    )

    preds = PREDS.clone().requires_grad_(True)

    with pytest.raises(RuntimeError):
        metric(preds, TARGET)


def test_estoi_rejects_autograd_tracked_inputs() -> None:
    """
    Test that ESTOI rejects tensors requiring gradients in grad mode.
    """
    metric = ESTOI(
        sample_rate=SAMPLE_RATE,
        reduction="mean",
    )

    preds = PREDS.clone().requires_grad_(True)

    with pytest.raises(RuntimeError):
        metric(preds, TARGET)


#############################################
# PESQ tests
#############################################

def test_pesq_forward_returns_scalar() -> None:
    """
    Test PESQ forward pass under mean reduction.
    """
    metric = PESQ(
        sample_rate=SAMPLE_RATE,
        mode="wb",
        reduction="mean",
    )

    check_metric_returns_scalar(metric, PREDS, TARGET)


def test_pesq_rejects_invalid_sample_rate() -> None:
    """
    Test that PESQ rejects unsupported sample rates.
    """
    with pytest.raises(ValueError):
        PESQ(
            sample_rate=44100,
            mode="wb",
            reduction="mean",
        )


def test_pesq_rejects_invalid_mode() -> None:
    """
    Test that PESQ rejects unsupported modes.
    """
    with pytest.raises(ValueError):
        PESQ(
            sample_rate=SAMPLE_RATE,
            mode="invalid",
            reduction="mean",
        )


def test_pesq_rejects_autograd_tracked_inputs() -> None:
    """
    Test that PESQ rejects tensors requiring gradients in grad mode.
    """
    metric = PESQ(
        sample_rate=SAMPLE_RATE,
        mode="wb",
        reduction="mean",
    )

    preds = PREDS.clone().requires_grad_(True)

    with pytest.raises(RuntimeError):
        metric(preds, TARGET)


#############################################
# DNSMOS tests
#############################################

def test_dnsmos_p808_forward_returns_scalar() -> None:
    """
    Test DNSMOS P.808 forward pass under mean reduction.
    """
    metric = DNSMOSP808(
        sample_rate=SAMPLE_RATE,
        reduction="mean",
    )

    check_metric_returns_scalar(metric, PREDS)


def test_dnsmos_sig_forward_returns_scalar() -> None:
    """
    Test DNSMOS SIG forward pass under mean reduction.
    """
    metric = DNSMOSSIG(
        sample_rate=SAMPLE_RATE,
        reduction="mean",
    )

    check_metric_returns_scalar(metric, PREDS)


def test_dnsmos_bak_forward_returns_scalar() -> None:
    """
    Test DNSMOS BAK forward pass under mean reduction.
    """
    metric = DNSMOSBAK(
        sample_rate=SAMPLE_RATE,
        reduction="mean",
    )

    check_metric_returns_scalar(metric, PREDS)


def test_dnsmos_ovrl_forward_returns_scalar() -> None:
    """
    Test DNSMOS OVRL forward pass under mean reduction.
    """
    metric = DNSMOSOVRL(
        sample_rate=SAMPLE_RATE,
        reduction="mean",
    )

    check_metric_returns_scalar(metric, PREDS)


def test_dnsmos_p835_forward_returns_three_scalars() -> None:
    """
    Test DNSMOS P.835 returns SIG, BAK, and OVRL scalar tensors.
    """
    metric = DNSMOSP835(
        sample_rate=SAMPLE_RATE,
        reduction="mean",
    )

    with torch.no_grad():
        values = metric(PREDS)

    assert isinstance(values, tuple)
    assert len(values) == 3

    for value in values:
        assert isinstance(value, torch.Tensor)
        assert value.ndim == 0


def test_dnsmos_rejects_autograd_tracked_inputs() -> None:
    """
    Test that DNSMOS rejects tensors requiring gradients in grad mode.
    """
    metric = DNSMOSP808(
        sample_rate=SAMPLE_RATE,
        reduction="mean",
    )

    preds = PREDS.clone().requires_grad_(True)

    with pytest.raises(RuntimeError):
        metric(preds)


#############################################
# Reduction tests
#############################################

def test_subjective_metrics_support_none_reduction() -> None:
    """
    Test that scalar-output subjective metrics preserve batch outputs with none reduction.
    """
    metrics_and_inputs = [
        (STOI(sample_rate=SAMPLE_RATE, reduction="none"), (PREDS, TARGET)),
        (ESTOI(sample_rate=SAMPLE_RATE, reduction="none"), (PREDS, TARGET)),
        (PESQ(sample_rate=SAMPLE_RATE, mode="wb", reduction="none"), (PREDS, TARGET)),
        (DNSMOSP808(sample_rate=SAMPLE_RATE, reduction="none"), (PREDS,)),
        (DNSMOSSIG(sample_rate=SAMPLE_RATE, reduction="none"), (PREDS,)),
        (DNSMOSBAK(sample_rate=SAMPLE_RATE, reduction="none"), (PREDS,)),
        (DNSMOSOVRL(sample_rate=SAMPLE_RATE, reduction="none"), (PREDS,)),
    ]

    for metric, inputs in metrics_and_inputs:
        check_metric_supports_none_reduction(metric, *inputs)


#############################################
# Validation tests
#############################################

def test_intrusive_subjective_metrics_reject_shape_mismatch() -> None:
    """
    Test that intrusive subjective metrics reject mismatching shapes.
    """
    metric = STOI(
        sample_rate=SAMPLE_RATE,
        reduction="mean",
    )

    wrong_target = torch.randn(BATCH_SIZE, CHANNELS, SIGNAL_LENGTH + 1)

    with pytest.raises(ValueError):
        metric(PREDS, wrong_target)


def test_intrusive_subjective_metrics_reject_multi_channel_input() -> None:
    """
    Test that intrusive subjective metrics reject multi-channel inputs.
    """
    metric = STOI(
        sample_rate=SAMPLE_RATE,
        reduction="mean",
    )

    multi_channel_preds = torch.randn(BATCH_SIZE, 2, SIGNAL_LENGTH)
    multi_channel_target = torch.randn(BATCH_SIZE, 2, SIGNAL_LENGTH)

    with pytest.raises(ValueError):
        metric(multi_channel_preds, multi_channel_target)


def test_non_intrusive_subjective_metrics_reject_multi_channel_input() -> None:
    """
    Test that non-intrusive subjective metrics reject multi-channel inputs.
    """
    metric = DNSMOSP808(
        sample_rate=SAMPLE_RATE,
        reduction="mean",
    )

    multi_channel_preds = torch.randn(BATCH_SIZE, 2, SIGNAL_LENGTH)

    with pytest.raises(ValueError):
        metric(multi_channel_preds)