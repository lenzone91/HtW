"""
Tests for the waveform dataset base class.

This file validates waveform-specific tensor conversion and normalization helpers.
"""

import pytest
import torch

from Project_1_SepFormer_TSE_HF.idea_1_project_setup.src.Datasets.waveform_base import (
    WaveformBaseDataset,
)


#############################################
# to_waveform_tensor
#############################################


def test_waveform_base_to_waveform_tensor_converts_1d_audio_to_single_channel() -> None:
    dataset = WaveformBaseDataset()

    waveform = dataset.to_waveform_tensor([1.0, 2.0, 3.0])

    assert waveform.shape == (1, 3)


def test_waveform_base_to_waveform_tensor_preserves_existing_channel_dimension() -> None:
    dataset = WaveformBaseDataset()

    waveform = dataset.to_waveform_tensor([[1.0, 2.0, 3.0]])

    assert waveform.shape == (1, 3)


def test_waveform_base_to_waveform_tensor_uses_canonical_dtype() -> None:
    dataset = WaveformBaseDataset(canonical_dtype=torch.float64)

    waveform = dataset.to_waveform_tensor([1.0, 2.0, 3.0])

    assert waveform.dtype == torch.float64


#############################################
# normalize_by_reference_peak
#############################################


def test_waveform_base_normalize_by_reference_peak() -> None:
    waveform = torch.tensor([[1.0, -2.0, 4.0]])
    reference = torch.tensor([[2.0, -8.0, 4.0]])

    normalized_waveform = WaveformBaseDataset.normalize_by_reference_peak(
        waveform=waveform,
        reference=reference,
        epsilon=0.0,
    )

    expected_waveform = torch.tensor([[0.125, -0.25, 0.5]])

    assert torch.allclose(normalized_waveform, expected_waveform)


def test_waveform_base_normalize_group_by_reference_peak() -> None:
    waveforms = {
        "mixture": torch.tensor([[2.0, -4.0]]),
        "target": torch.tensor([[1.0, -2.0]]),
    }

    normalized_waveforms = WaveformBaseDataset.normalize_group_by_reference_peak(
        waveforms=waveforms,
        reference_key="mixture",
        epsilon=0.0,
    )

    assert torch.allclose(
        normalized_waveforms["mixture"],
        torch.tensor([[0.5, -1.0]]),
    )
    assert torch.allclose(
        normalized_waveforms["target"],
        torch.tensor([[0.25, -0.5]]),
    )


def test_waveform_base_normalize_group_by_reference_peak_rejects_missing_reference_key() -> None:
    waveforms = {
        "target": torch.tensor([[1.0, -2.0]]),
    }

    with pytest.raises(KeyError):
        WaveformBaseDataset.normalize_group_by_reference_peak(
            waveforms=waveforms,
            reference_key="mixture",
        )