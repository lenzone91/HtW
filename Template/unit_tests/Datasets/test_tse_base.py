"""
Tests for the TSE dataset base classes.

This file validates TSE sample conventions and clue-pairing behavior.
"""

from typing import Any

import pytest

from Project_1_SepFormer_TSE_HF.idea_1_project_setup.src.Datasets.base import (
    BaseDataset,
)
from Project_1_SepFormer_TSE_HF.idea_1_project_setup.src.Datasets.clue_base import (
    BaseClueDataset,
)
from Project_1_SepFormer_TSE_HF.idea_1_project_setup.src.Datasets.tse_base import (
    BaseTSEDataset,
    BaseTSEDatasetWithClue,
    PairedTSEDatasetWithClue,
)


#############################################
# Dummy datasets
#############################################


class DummyBaseDataset(BaseDataset):
    def __init__(self, samples: list[dict[str, Any]]) -> None:
        super().__init__()
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.samples[index]


class DummyClueDataset(BaseClueDataset):
    def __init__(self, samples: dict[str, dict[str, Any]]) -> None:
        super().__init__()
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, key: str) -> dict[str, Any]:
        return self.samples[key]


#############################################
# BaseTSEDataset
#############################################


def test_base_tse_dataset_build_tse_sample() -> None:
    sample = BaseTSEDataset.build_tse_sample(
        mixture="mixture_data",
        target="target_data",
        clue=None,
        metadata={"sample_id": "sample_0"},
    )

    assert sample == {
        "mixture": "mixture_data",
        "target": "target_data",
        "clue": None,
        "metadata": {"sample_id": "sample_0"},
    }


def test_base_tse_dataset_check_sample_accepts_valid_tse_sample() -> None:
    dataset = BaseTSEDataset()

    sample = BaseTSEDataset.build_tse_sample(
        mixture="mixture_data",
        target="target_data",
        clue=None,
        metadata={},
    )

    dataset.check_sample(sample)


def test_base_tse_dataset_check_sample_rejects_missing_required_key() -> None:
    dataset = BaseTSEDataset()

    sample = {
        "mixture": "mixture_data",
        "target": "target_data",
        "metadata": {},
    }

    with pytest.raises(KeyError):
        dataset.check_sample(sample)


#############################################
# BaseTSEDatasetWithClue
#############################################


def test_base_tse_dataset_with_clue_build_tse_sample_rejects_missing_clue() -> None:
    with pytest.raises(ValueError):
        BaseTSEDatasetWithClue.build_tse_sample(
            mixture="mixture_data",
            target="target_data",
            clue=None,
            metadata={},
        )


#############################################
# PairedTSEDatasetWithClue
#############################################


def test_paired_tse_dataset_with_clue_len_matches_base_dataset() -> None:
    base_dataset = DummyBaseDataset(
        samples=[
            BaseDataset.build_sample(
                input="mixture_0",
                target="target_0",
                metadata={"speaker_id": "speaker_0"},
            ),
            BaseDataset.build_sample(
                input="mixture_1",
                target="target_1",
                metadata={"speaker_id": "speaker_1"},
            ),
        ],
    )
    clue_dataset = DummyClueDataset(samples={})

    dataset = PairedTSEDatasetWithClue(
        base_dataset=base_dataset,
        clue_dataset=clue_dataset,
        clue_lookup_key="speaker_id",
    )

    assert len(dataset) == 2


def test_paired_tse_dataset_with_clue_getitem_pairs_base_and_clue_samples() -> None:
    base_dataset = DummyBaseDataset(
        samples=[
            BaseDataset.build_sample(
                input="mixture_0",
                target="target_0",
                metadata={"speaker_id": "speaker_0", "sample_id": "sample_0"},
            ),
        ],
    )
    clue_dataset = DummyClueDataset(
        samples={
            "speaker_0": BaseClueDataset.build_clue_sample(
                clue="clue_0",
                metadata={"clue_id": "clue_sample_0"},
            ),
        },
    )

    dataset = PairedTSEDatasetWithClue(
        base_dataset=base_dataset,
        clue_dataset=clue_dataset,
        clue_lookup_key="speaker_id",
    )

    sample = dataset[0]

    assert sample == {
        "mixture": "mixture_0",
        "target": "target_0",
        "clue": "clue_0",
        "metadata": {
            "speaker_id": "speaker_0",
            "sample_id": "sample_0",
            "clue_metadata": {"clue_id": "clue_sample_0"},
        },
    }


def test_paired_tse_dataset_with_clue_get_clue_key_rejects_missing_lookup_key() -> None:
    dataset = PairedTSEDatasetWithClue(
        base_dataset=DummyBaseDataset(samples=[]),
        clue_dataset=DummyClueDataset(samples={}),
        clue_lookup_key="speaker_id",
    )

    with pytest.raises(KeyError):
        dataset.get_clue_key(base_metadata={"sample_id": "sample_0"})


def test_paired_tse_dataset_with_clue_build_metadata_nests_clue_metadata() -> None:
    metadata = PairedTSEDatasetWithClue.build_metadata(
        base_metadata={"speaker_id": "speaker_0"},
        clue_metadata={"clue_id": "clue_sample_0"},
    )

    assert metadata == {
        "speaker_id": "speaker_0",
        "clue_metadata": {"clue_id": "clue_sample_0"},
    }