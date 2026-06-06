"""
Tests for the generic Lightning DataModule base class.

This file validates phase consistency checks and phase-specific DataLoader creation.
"""

from typing import Any

import pytest
import torch
from torch.utils.data import Dataset, DataLoader

from Project_1_SepFormer_TSE_HF.idea_1_project_setup.src.DataModules.base import (
    BaseDataModule,
)


#############################################
# Dummy objects
#############################################


class DummyDataset(Dataset):
    def __init__(self, values: list[int]) -> None:
        self.values = values

    def __len__(self) -> int:
        return len(self.values)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return {"value": self.values[index]}


class DummyCollator:
    def __call__(self, samples: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        values = [sample["value"] for sample in samples]
        return {"value": torch.tensor(values)}


def make_valid_datamodule() -> BaseDataModule:
    return BaseDataModule(
        datasets={
            "train": DummyDataset([0, 1, 2, 3]),
            "val": DummyDataset([4, 5]),
            "test": DummyDataset([6, 7]),
        },
        collators={
            "train": DummyCollator(),
            "val": DummyCollator(),
            "test": DummyCollator(),
        },
        dataloader_configs={
            "train": {"batch_size": 2, "shuffle": False},
            "val": {"batch_size": 2, "shuffle": False},
            "test": {"batch_size": 2, "shuffle": False},
        },
    )


#############################################
# Phase consistency
#############################################


def test_base_datamodule_rejects_missing_collator_for_existing_dataset() -> None:
    with pytest.raises(KeyError):
        BaseDataModule(
            datasets={"train": DummyDataset([0, 1])},
            collators={},
            dataloader_configs={"train": {"batch_size": 2}},
        )


def test_base_datamodule_rejects_none_collator_for_existing_dataset() -> None:
    with pytest.raises(ValueError):
        BaseDataModule(
            datasets={"train": DummyDataset([0, 1])},
            collators={"train": None},
            dataloader_configs={"train": {"batch_size": 2}},
        )


def test_base_datamodule_rejects_missing_dataloader_config_for_existing_dataset() -> None:
    with pytest.raises(KeyError):
        BaseDataModule(
            datasets={"train": DummyDataset([0, 1])},
            collators={"train": DummyCollator()},
            dataloader_configs={},
        )


def test_base_datamodule_rejects_unknown_phase() -> None:
    datamodule = make_valid_datamodule()

    with pytest.raises(KeyError):
        datamodule.make_dataloader("predict")


#############################################
# DataLoader creation
#############################################


def test_base_datamodule_make_dataloader_returns_none_when_dataset_is_none() -> None:
    datamodule = BaseDataModule(
        datasets={"train": None},
        collators={},
        dataloader_configs={},
    )

    dataloader = datamodule.make_dataloader("train")

    assert dataloader is None


def test_base_datamodule_train_dataloader_returns_dataloader() -> None:
    datamodule = make_valid_datamodule()

    dataloader = datamodule.train_dataloader()

    assert isinstance(dataloader, DataLoader)


def test_base_datamodule_val_dataloader_returns_dataloader() -> None:
    datamodule = make_valid_datamodule()

    dataloader = datamodule.val_dataloader()

    assert isinstance(dataloader, DataLoader)


def test_base_datamodule_test_dataloader_returns_dataloader() -> None:
    datamodule = make_valid_datamodule()

    dataloader = datamodule.test_dataloader()

    assert isinstance(dataloader, DataLoader)


#############################################
# Phase-specific behavior
#############################################


def test_base_datamodule_uses_phase_specific_collator() -> None:
    datamodule = make_valid_datamodule()

    batch = next(iter(datamodule.train_dataloader()))

    assert isinstance(batch, dict)
    assert torch.equal(batch["value"], torch.tensor([0, 1]))


def test_base_datamodule_uses_phase_specific_dataloader_config() -> None:
    datamodule = BaseDataModule(
        datasets={"train": DummyDataset([0, 1, 2, 3])},
        collators={"train": DummyCollator()},
        dataloader_configs={"train": {"batch_size": 3, "shuffle": False}},
    )

    batch = next(iter(datamodule.train_dataloader()))

    assert torch.equal(batch["value"], torch.tensor([0, 1, 2]))