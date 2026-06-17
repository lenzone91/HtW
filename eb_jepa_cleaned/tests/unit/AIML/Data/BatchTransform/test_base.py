"""
Unit tests for AIML.Data.BatchTransform.base.BaseBatchTransform.

The shared dict -> dict contract for both DataAugmentation and DataAdaptation.
"""

import pytest
import torch

from eb_jepa_cleaned.AIML.Data.BatchTransform.base import (
    BaseBatchTransform,
)


class AddKey(BaseBatchTransform):
    def transform(self, batch):
        return {**batch, "added": True}


class BadOutput(BaseBatchTransform):
    def transform(self, batch):
        return ["not", "a", "dict"]


def test_call_validates_and_transforms():
    assert AddKey()({"x": 1}) == {"x": 1, "added": True}


def test_call_rejects_non_dict_input():
    with pytest.raises(TypeError, match="input batch to be a dict"):
        AddKey()([1, 2])


def test_call_rejects_non_dict_output():
    with pytest.raises(TypeError, match="output batch to be a dict"):
        BadOutput()({"x": 1})


def test_check_required_key():
    transform = AddKey()

    transform.check_required_key({"a": 1}, "a")
    with pytest.raises(KeyError, match="expected key 'b'"):
        transform.check_required_key({"a": 1}, "b")


def test_check_is_batched_tensor():
    transform = AddKey()

    transform.check_is_batched_tensor(torch.zeros(2, 3), name="x", min_ndim=2)

    with pytest.raises(ValueError, match="at least 2 dimensions"):
        transform.check_is_batched_tensor(torch.zeros(3), name="x", min_ndim=2)

    with pytest.raises(TypeError, match="to be a torch.Tensor"):
        transform.check_is_batched_tensor([1, 2], name="x")
