"""
Pytest test module for Metrics.objective.

Pytest automatically discovers and runs functions whose names start with
``test_``. Therefore, no explicit test decorator is needed here.
"""

import pytest
import torch

from Project_1_SepFormer_TSE_HF.idea_1_project_setup.src.Metrics.objective import (
    DTW,
    ItakuraSaitoDivergence,
    LPError,
    LPNorm,
    LSD,
    SDSDR,
    SISAR,
    SISDR,
    SISIR,
    SNR,
    SpectralKLDivergence,
)


#############################################
# Test data
#############################################

BATCH_SIZE = 2
CHANNELS = 1
SIGNAL_LENGTH = 1024

PREDS = torch.randn(BATCH_SIZE, CHANNELS, SIGNAL_LENGTH)
TARGET = torch.randn(BATCH_SIZE, CHANNELS, SIGNAL_LENGTH)
MIXTURE = TARGET + torch.randn(BATCH_SIZE, CHANNELS, SIGNAL_LENGTH)


#############################################
# Shared checks
#############################################

def check_metric_returns_scalar(metric, *args) -> None:
    """
    Check that a metric returns a scalar tensor under mean reduction.
    """
    value = metric(*args)

    assert isinstance(value, torch.Tensor)
    assert value.ndim == 0


def check_metric_supports_none_reduction(metric, *args) -> None:
    """
    Check that a metric returns one value per batch item under none reduction.
    """
    value = metric(*args)

    assert isinstance(value, torch.Tensor)
    assert value.shape == torch.Size([BATCH_SIZE])


#############################################
# Intrusive objective metric tests
#############################################

def test_snr_forward_returns_scalar() -> None:
    """
    Test SNR forward pass under mean reduction.
    """
    metric = SNR(reduction="mean")

    check_metric_returns_scalar(metric, PREDS, TARGET)


def test_sisdr_forward_returns_scalar() -> None:
    """
    Test SI-SDR forward pass under mean reduction.
    """
    metric = SISDR(reduction="mean")

    check_metric_returns_scalar(metric, PREDS, TARGET)


def test_sdsdr_forward_returns_scalar() -> None:
    """
    Test SD-SDR forward pass under mean reduction.
    """
    metric = SDSDR(reduction="mean")

    check_metric_returns_scalar(metric, PREDS, TARGET)


def test_lpnorm_forward_returns_scalar() -> None:
    """
    Test Lp norm forward pass under mean reduction.
    """
    metric = LPNorm(reduction="mean")

    check_metric_returns_scalar(metric, PREDS)


def test_lperror_forward_returns_scalar() -> None:
    """
    Test Lp error forward pass under mean reduction.
    """
    metric = LPError(reduction="mean")

    check_metric_returns_scalar(metric, PREDS, TARGET)


#############################################
# Scale-invariant decomposition tests
#############################################

def test_sisir_forward_returns_scalar() -> None:
    """
    Test SI-SIR forward pass under mean reduction.
    """
    metric = SISIR(reduction="mean")

    check_metric_returns_scalar(metric, PREDS, TARGET, MIXTURE)


def test_sisar_forward_returns_scalar() -> None:
    """
    Test SI-SAR forward pass under mean reduction.
    """
    metric = SISAR(reduction="mean")

    check_metric_returns_scalar(metric, PREDS, TARGET, MIXTURE)


#############################################
# Spectral objective metric tests
#############################################

def test_lsd_forward_returns_scalar() -> None:
    """
    Test LSD forward pass under mean reduction.
    """
    metric = LSD(
        n_fft=128,
        hop_length=64,
        win_length=128,
        reduction="mean",
    )

    check_metric_returns_scalar(metric, PREDS, TARGET)


def test_spectral_kl_forward_returns_scalar() -> None:
    """
    Test spectral KL divergence forward pass under mean reduction.
    """
    metric = SpectralKLDivergence(
        n_fft=128,
        hop_length=64,
        win_length=128,
        reduction="mean",
    )

    check_metric_returns_scalar(metric, PREDS, TARGET)


def test_itakura_saito_forward_returns_scalar() -> None:
    """
    Test Itakura-Saito divergence forward pass under mean reduction.
    """
    metric = ItakuraSaitoDivergence(
        n_fft=128,
        hop_length=64,
        win_length=128,
        reduction="mean",
    )

    check_metric_returns_scalar(metric, PREDS, TARGET)


#############################################
# Evaluation-only objective metric tests
#############################################

def test_dtw_forward_returns_scalar_under_no_grad() -> None:
    """
    Test DTW forward pass under no_grad because DTW is evaluation-only.
    """
    metric = DTW(reduction="mean")

    with torch.no_grad():
        check_metric_returns_scalar(metric, PREDS, TARGET)


def test_dtw_rejects_autograd_tracked_inputs() -> None:
    """
    Test that DTW rejects tensors requiring gradients in grad mode.
    """
    metric = DTW(reduction="mean")

    preds = PREDS.clone().requires_grad_(True)

    with pytest.raises(RuntimeError):
        metric(preds, TARGET)


#############################################
# Reduction tests
#############################################

def test_objective_metrics_support_none_reduction() -> None:
    """
    Test that lightweight objective metrics preserve batch outputs with none reduction.
    """
    metrics_and_inputs = [
        (SNR(reduction="none"), (PREDS, TARGET)),
        (SISDR(reduction="none"), (PREDS, TARGET)),
        (SDSDR(reduction="none"), (PREDS, TARGET)),
        (LPNorm(reduction="none"), (PREDS,)),
        (LPError(reduction="none"), (PREDS, TARGET)),
        (SISIR(reduction="none"), (PREDS, TARGET, MIXTURE)),
        (SISAR(reduction="none"), (PREDS, TARGET, MIXTURE)),
    ]

    for metric, inputs in metrics_and_inputs:
        check_metric_supports_none_reduction(metric, *inputs)


#############################################
# Validation tests
#############################################

def test_intrusive_metrics_reject_shape_mismatch() -> None:
    """
    Test that intrusive metrics reject mismatching prediction and target shapes.
    """
    metric = SDSDR()
    wrong_target = torch.randn(BATCH_SIZE, CHANNELS, SIGNAL_LENGTH + 1)

    with pytest.raises(ValueError):
        metric(PREDS, wrong_target)


def test_intrusive_metrics_reject_multi_channel_input() -> None:
    """
    Test that intrusive waveform metrics reject multi-channel inputs.
    """
    metric = SDSDR()
    multi_channel_preds = torch.randn(BATCH_SIZE, 2, SIGNAL_LENGTH)
    multi_channel_target = torch.randn(BATCH_SIZE, 2, SIGNAL_LENGTH)

    with pytest.raises(ValueError):
        metric(multi_channel_preds, multi_channel_target)