"""
Pytest test module for DataProcessing.base.

Pytest automatically discovers and runs functions whose names start with
``test_``. Therefore, no explicit test decorator is needed here.
"""

import pytest
import torch

from Project_1_SepFormer_TSE_HF.idea_1_project_setup.src.DataProcessing.base import (
    BaseBatchTransform,
    BaseCollator,
)


#############################################
# Dummy objects
#############################################

class IdentityTransform(BaseBatchTransform):
    """
    Minimal transform returning the input batch unchanged.
    """

    def transform(self, batch: dict) -> dict:
        return batch


class InvalidOutputTransform(BaseBatchTransform):
    """
    Transform returning an invalid non-dict output.
    """

    def transform(self, batch: dict) -> list:
        return ["invalid"]


class AddFlagTransform(BaseBatchTransform):
    """
    Transform recording its application order in the batch.
    """

    def __init__(self, flag: str) -> None:
        self.flag = flag

    def transform(self, batch: dict) -> dict:
        batch["flags"].append(self.flag)
        return batch


class DummyCollator(BaseCollator):
    """
    Minimal collator returning a valid batch dictionary.
    """

    def collate_samples(self, samples: list[dict]) -> dict:
        return {
            "samples": samples,
            "flags": [],
        }


class InvalidOutputCollator(BaseCollator):
    """
    Collator returning an invalid non-dict batch.
    """

    def collate_samples(self, samples: list[dict]) -> list:
        return ["invalid"]


#############################################
# BaseBatchTransform tests
#############################################

def test_base_batch_transform_rejects_non_dict_input() -> None:
    """
    Test that transforms require dictionary inputs.
    """
    transform = IdentityTransform()

    with pytest.raises(TypeError):
        transform(["invalid"])


def test_base_batch_transform_rejects_non_dict_output() -> None:
    """
    Test that transforms must return dictionary outputs.
    """
    transform = InvalidOutputTransform()

    with pytest.raises(TypeError):
        transform({})


def test_base_batch_transform_checks_required_key() -> None:
    """
    Test required-key validation.
    """
    transform = IdentityTransform()

    with pytest.raises(KeyError):
        transform.check_required_key(
            batch={},
            key="missing",
        )


def test_base_batch_transform_checks_tensor() -> None:
    """
    Test tensor validation.
    """
    transform = IdentityTransform()

    transform.check_is_tensor(
        value=torch.tensor(1.0),
        name="value",
    )

    with pytest.raises(TypeError):
        transform.check_is_tensor(
            value=1.0,
            name="value",
        )


def test_base_batch_transform_checks_batched_tensor() -> None:
    """
    Test batched tensor validation.
    """
    transform = IdentityTransform()

    transform.check_is_batched_tensor(
        value=torch.randn(2, 8),
        name="value",
    )

    with pytest.raises(ValueError):
        transform.check_is_batched_tensor(
            value=torch.randn(8),
            name="value",
        )


#############################################
# BaseCollator input tests
#############################################

def test_base_collator_rejects_non_list_samples() -> None:
    """
    Test that collators require a list of samples.
    """
    collator = DummyCollator()

    with pytest.raises(TypeError):
        collator({"sample": 1})


def test_base_collator_rejects_empty_sample_list() -> None:
    """
    Test that collators reject empty sample lists.
    """
    collator = DummyCollator()

    with pytest.raises(ValueError):
        collator([])


def test_base_collator_rejects_non_dict_samples() -> None:
    """
    Test that every sample must be a dictionary.
    """
    collator = DummyCollator()

    with pytest.raises(TypeError):
        collator([{"valid": 1}, "invalid"])


def test_base_collator_rejects_non_dict_collated_batch() -> None:
    """
    Test that collate_samples must return a dictionary.
    """
    collator = InvalidOutputCollator()

    with pytest.raises(TypeError):
        collator([{"sample": 1}])


#############################################
# BaseCollator transform pipeline tests
#############################################

def test_base_collator_applies_transforms_in_order() -> None:
    """
    Test that registered transforms are applied in order.
    """
    collator = DummyCollator(
        transforms=[
            AddFlagTransform("first"),
            AddFlagTransform("second"),
        ],
    )

    batch = collator([{"sample": 1}])

    assert batch["flags"] == ["first", "second"]