"""
Tests for the generic dataset base class.

This file validates project-level sample conventions and helper methods.
"""

import pytest
import torch

from Project_1_SepFormer_TSE_HF.idea_1_project_setup.src.Datasets.base import (
    BaseDataset,
)


#############################################
# to_tensor
#############################################


def test_base_dataset_to_tensor_uses_canonical_dtype() -> None:
    dataset = BaseDataset(canonical_dtype=torch.float64)

    tensor = dataset.to_tensor([1.0, 2.0, 3.0])

    assert isinstance(tensor, torch.Tensor)
    assert tensor.dtype == torch.float64


#############################################
# build_sample
#############################################


def test_base_dataset_build_sample_with_metadata() -> None:
    sample = BaseDataset.build_sample(
        input="input_data",
        target="target_data",
        metadata={"sample_id": "sample_0"},
    )

    assert sample == {
        "input": "input_data",
        "target": "target_data",
        "metadata": {"sample_id": "sample_0"},
    }


def test_base_dataset_build_sample_defaults_to_empty_metadata() -> None:
    sample = BaseDataset.build_sample(
        input="input_data",
        target="target_data",
    )

    assert sample["metadata"] == {}


#############################################
# check_required_keys
#############################################


def test_base_dataset_check_required_keys_accepts_valid_sample() -> None:
    dataset = BaseDataset()

    sample = BaseDataset.build_sample(
        input="input_data",
        target="target_data",
        metadata={},
    )

    dataset.check_required_keys(sample)


def test_base_dataset_check_required_keys_rejects_missing_key() -> None:
    dataset = BaseDataset()

    sample = {
        "input": "input_data",
        "metadata": {},
    }

    with pytest.raises(KeyError):
        dataset.check_required_keys(sample)


#############################################
# check_sample
#############################################


def test_base_dataset_check_sample_rejects_non_dict_sample() -> None:
    dataset = BaseDataset()

    with pytest.raises(TypeError):
        dataset.check_sample(["input_data", "target_data"])


def test_base_dataset_check_sample_rejects_non_dict_metadata() -> None:
    dataset = BaseDataset()

    sample = BaseDataset.build_sample(
        input="input_data",
        target="target_data",
        metadata=None,
    )
    sample["metadata"] = None

    with pytest.raises(TypeError):
        dataset.check_sample(sample)