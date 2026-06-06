"""
Tests for the clue dataset base class.

This file validates clue-specific sample conventions and lookup key checks.
"""

import pytest

from Project_1_SepFormer_TSE_HF.idea_1_project_setup.src.Datasets.clue_base import (
    BaseClueDataset,
)


#############################################
# build_clue_sample
#############################################


def test_clue_base_dataset_build_clue_sample_with_metadata() -> None:
    sample = BaseClueDataset.build_clue_sample(
        clue="clue_data",
        metadata={"speaker_id": "speaker_0"},
    )

    assert sample == {
        "clue": "clue_data",
        "metadata": {"speaker_id": "speaker_0"},
    }


def test_clue_base_dataset_build_clue_sample_defaults_to_empty_metadata() -> None:
    sample = BaseClueDataset.build_clue_sample(
        clue="clue_data",
    )

    assert sample["metadata"] == {}


#############################################
# check_sample
#############################################


def test_clue_base_dataset_check_sample_accepts_valid_clue_sample() -> None:
    dataset = BaseClueDataset()

    sample = BaseClueDataset.build_clue_sample(
        clue="clue_data",
        metadata={},
    )

    dataset.check_sample(sample)


def test_clue_base_dataset_check_sample_rejects_non_dict_sample() -> None:
    dataset = BaseClueDataset()

    with pytest.raises(TypeError):
        dataset.check_sample(["clue_data"])


def test_clue_base_dataset_check_sample_rejects_missing_required_key() -> None:
    dataset = BaseClueDataset()

    sample = {
        "metadata": {},
    }

    with pytest.raises(KeyError):
        dataset.check_sample(sample)


def test_clue_base_dataset_check_sample_rejects_non_dict_metadata() -> None:
    dataset = BaseClueDataset()

    sample = BaseClueDataset.build_clue_sample(
        clue="clue_data",
        metadata={},
    )
    sample["metadata"] = None

    with pytest.raises(TypeError):
        dataset.check_sample(sample)


#############################################
# check_key
#############################################


def test_clue_base_dataset_check_key_accepts_valid_key() -> None:
    dataset = BaseClueDataset()

    dataset.check_key("speaker_0")


def test_clue_base_dataset_check_key_rejects_none() -> None:
    dataset = BaseClueDataset()

    with pytest.raises(ValueError):
        dataset.check_key(None)