"""
Tests for the toy source-separation dataset.

This file validates loading, formatting, and normalization on synthetic wav data.
"""

from pathlib import Path

import pytest
import torch
from scipy.io import wavfile

from Project_1_SepFormer_TSE_HF.idea_1_project_setup.src.Datasets.toy_dataset import (
    SourceSeparationDataset,
)


#############################################
# Helpers
#############################################


def create_toy_source_separation_folder(
    tmp_path: Path,
) -> Path:
    dataset_folder = tmp_path / "toy_source_separation"
    dataset_folder.mkdir()

    sample_folder = dataset_folder / "sample_0"
    sample_folder.mkdir()

    sample_rate = 8000

    voice = torch.tensor([0.25, -0.25, 0.50, -0.50], dtype=torch.float32)
    noise = torch.tensor([0.10, -0.10, 0.20, -0.20], dtype=torch.float32)
    mixture = voice + noise

    wavfile.write(sample_folder / "voice.wav", sample_rate, voice.numpy())
    wavfile.write(sample_folder / "noise.wav", sample_rate, noise.numpy())
    wavfile.write(sample_folder / "mix_5.0.wav", sample_rate, mixture.numpy())

    return dataset_folder


#############################################
# Initialization
#############################################


def test_source_separation_dataset_rejects_invalid_folder(tmp_path: Path) -> None:
    invalid_folder = tmp_path / "missing_folder"

    with pytest.raises(ValueError):
        SourceSeparationDataset(path_to_folder=str(invalid_folder))


def test_source_separation_dataset_initializes_from_toy_folder(tmp_path: Path) -> None:
    dataset_folder = create_toy_source_separation_folder(tmp_path)

    dataset = SourceSeparationDataset(path_to_folder=str(dataset_folder))

    assert isinstance(dataset, SourceSeparationDataset)


#############################################
# Static helpers
#############################################


def test_source_separation_dataset_extract_snr_from_mix_filename() -> None:
    snr = SourceSeparationDataset.extract_snr_from_mix_filename("mix_12.5.wav")

    assert snr == 12.5


#############################################
# Dataset protocol
#############################################


def test_source_separation_dataset_len_matches_number_of_data_ids(tmp_path: Path) -> None:
    dataset_folder = create_toy_source_separation_folder(tmp_path)

    dataset = SourceSeparationDataset(path_to_folder=str(dataset_folder))

    assert len(dataset) == 1


def test_source_separation_dataset_getitem_returns_tse_sample(tmp_path: Path) -> None:
    dataset_folder = create_toy_source_separation_folder(tmp_path)

    dataset = SourceSeparationDataset(path_to_folder=str(dataset_folder))

    sample = dataset[0]

    assert set(sample.keys()) == {"mixture", "target", "clue", "metadata"}
    assert sample["clue"] is None
    assert isinstance(sample["metadata"], dict)


def test_source_separation_dataset_getitem_returns_single_channel_waveforms(
    tmp_path: Path,
) -> None:
    dataset_folder = create_toy_source_separation_folder(tmp_path)

    dataset = SourceSeparationDataset(path_to_folder=str(dataset_folder))

    sample = dataset[0]

    assert sample["mixture"].shape == (1, 4)
    assert sample["target"].shape == (1, 4)


def test_source_separation_dataset_getitem_extracts_snr_metadata(tmp_path: Path) -> None:
    dataset_folder = create_toy_source_separation_folder(tmp_path)

    dataset = SourceSeparationDataset(path_to_folder=str(dataset_folder))

    sample = dataset[0]

    assert sample["metadata"]["snr"] == 5.0


def test_source_separation_dataset_supports_load_on_demand(tmp_path: Path) -> None:
    dataset_folder = create_toy_source_separation_folder(tmp_path)

    dataset = SourceSeparationDataset(
        path_to_folder=str(dataset_folder),
        load_on_demand=True,
    )

    sample = dataset[0]

    assert dataset.data is None
    assert set(sample.keys()) == {"mixture", "target", "clue", "metadata"}


#############################################
# Normalization
#############################################


def test_source_separation_dataset_normalizes_by_mixture_peak(tmp_path: Path) -> None:
    dataset_folder = create_toy_source_separation_folder(tmp_path)

    dataset = SourceSeparationDataset(
        path_to_folder=str(dataset_folder),
        normalize_by_mixture_peak=True,
    )

    sample = dataset[0]

    assert torch.max(torch.abs(sample["mixture"])) <= 1.0