"""
Unit tests for AIML.Data.Datasets.base.BaseDataset.

Covers the generic sample convention and shared validation/conversion helpers.
"""

import pytest
import torch

from eb_jepa_cleaned.AIML.Data.Datasets.base import (
    BaseDataset,
)


def test_build_sample_shape_and_default_metadata():
    sample = BaseDataset.build_sample(input=1, target=2)

    assert sample == {"input": 1, "target": 2, "metadata": {}}


def test_build_sample_keeps_given_metadata():
    sample = BaseDataset.build_sample(input=1, target=2, metadata={"k": "v"})

    assert sample["metadata"] == {"k": "v"}


def test_to_tensor_uses_canonical_dtype():
    dataset = BaseDataset(canonical_dtype=torch.float64)

    tensor = dataset.to_tensor([1.0, 2.0])

    assert tensor.dtype == torch.float64
    assert torch.equal(tensor, torch.tensor([1.0, 2.0], dtype=torch.float64))


def test_check_sample_accepts_valid_sample():
    dataset = BaseDataset()

    dataset.check_sample({"input": 1, "target": 2, "metadata": {}})


def test_check_sample_rejects_non_dict():
    dataset = BaseDataset()

    with pytest.raises(TypeError, match="Expected sample to be a dict"):
        dataset.check_sample([1, 2])


def test_check_sample_rejects_missing_keys():
    dataset = BaseDataset()

    with pytest.raises(KeyError, match="Missing sample keys"):
        dataset.check_sample({"input": 1, "metadata": {}})


def test_check_sample_rejects_non_dict_metadata():
    dataset = BaseDataset()

    with pytest.raises(TypeError, match="metadata.* to be a dict"):
        dataset.check_sample({"input": 1, "target": 2, "metadata": "nope"})
