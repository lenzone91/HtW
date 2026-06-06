"""
Pytest test module for DataProcessing.collators.

Pytest automatically discovers and runs functions whose names start with
``test_``. Therefore, no explicit test decorator is needed here.
"""

import pytest
import torch

from Project_1_SepFormer_TSE_HF.idea_1_project_setup.src.DataProcessing.base import (
    BaseBatchTransform,
)
from Project_1_SepFormer_TSE_HF.idea_1_project_setup.src.DataProcessing.collators import (
    TSEWaveformCollator,
)


#############################################
# Dummy transform
#############################################

class AddBatchFlagTransform(BaseBatchTransform):
    """
    Add a flag after collation to check transform application.
    """

    def transform(self, batch: dict) -> dict:
        batch["was_transformed"] = True
        return batch


#############################################
# Test data
#############################################

def make_sample(index: int, clue=None) -> dict:
    """
    Build one toy TSE waveform sample.
    """
    return {
        "mixture": torch.full((1, 8), float(index)),
        "target": torch.full((1, 8), float(index + 1)),
        "clue": clue,
        "metadata": {
            "index": index,
        },
    }


def make_samples() -> list[dict]:
    """
    Build a toy sample list.
    """
    return [
        make_sample(0),
        make_sample(1),
    ]


#############################################
# Collation tests
#############################################

def test_tse_waveform_collator_stacks_mixture_and_target() -> None:
    """
    Test that mixture and target tensors are stacked along batch dimension.
    """
    collator = TSEWaveformCollator()

    batch = collator(make_samples())

    assert batch["mixture"].shape == torch.Size([2, 1, 8])
    assert batch["target"].shape == torch.Size([2, 1, 8])


def test_tse_waveform_collator_returns_metadata_list() -> None:
    """
    Test that metadata entries are preserved as a list of dictionaries.
    """
    collator = TSEWaveformCollator()

    batch = collator(make_samples())

    assert batch["metadata"] == [
        {"index": 0},
        {"index": 1},
    ]


def test_tse_waveform_collator_returns_none_when_all_clues_are_none() -> None:
    """
    Test that clue is None when all samples have no clue.
    """
    collator = TSEWaveformCollator()

    batch = collator(make_samples())

    assert batch["clue"] is None


def test_tse_waveform_collator_stacks_tensor_clues() -> None:
    """
    Test that tensor clues with matching shapes are stacked.
    """
    samples = [
        make_sample(0, clue=torch.tensor([0.0, 1.0])),
        make_sample(1, clue=torch.tensor([2.0, 3.0])),
    ]

    collator = TSEWaveformCollator()

    batch = collator(samples)

    assert torch.equal(
        batch["clue"],
        torch.tensor(
            [
                [0.0, 1.0],
                [2.0, 3.0],
            ]
        ),
    )


def test_tse_waveform_collator_keeps_mixed_clues_as_list() -> None:
    """
    Test that mixed clue objects are kept as a list.
    """
    samples = [
        make_sample(0, clue=torch.tensor([0.0])),
        make_sample(1, clue="text_clue"),
    ]

    collator = TSEWaveformCollator()

    batch = collator(samples)

    assert isinstance(batch["clue"], list)
    assert len(batch["clue"]) == 2


def test_tse_waveform_collator_applies_transforms_after_collation() -> None:
    """
    Test that batch transforms are applied after sample collation.
    """
    collator = TSEWaveformCollator(
        transforms=[
            AddBatchFlagTransform(),
        ],
    )

    batch = collator(make_samples())

    assert batch["was_transformed"] is True


#############################################
# Failure tests
#############################################

def test_tse_waveform_collator_rejects_missing_mixture_key() -> None:
    """
    Test that missing mixture keys fail explicitly.
    """
    samples = make_samples()
    samples[0].pop("mixture")

    collator = TSEWaveformCollator()

    with pytest.raises(KeyError):
        collator(samples)


def test_tse_waveform_collator_rejects_missing_target_key() -> None:
    """
    Test that missing target keys fail explicitly.
    """
    samples = make_samples()
    samples[0].pop("target")

    collator = TSEWaveformCollator()

    with pytest.raises(KeyError):
        collator(samples)