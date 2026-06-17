"""
Unit tests for AIML.Data.DataModules.base.BaseDataModule.

Exercises phase storage, DataLoader construction, and phase-consistency checks
with already-built (dummy) datasets and collators.
"""

import pytest
from torch.utils.data import DataLoader, Dataset

from src.AIML.Data.Collators.base import (
    BaseCollator,
)
from src.AIML.Data.DataModules.base import (
    BaseDataModule,
)


class TinyDataset(Dataset):
    def __init__(self, n=4):
        self.n = n

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        return {"input": i, "target": i, "metadata": {}}


class ListCollator(BaseCollator):
    def collate_samples(self, samples):
        return {"input": [s["input"] for s in samples]}


def make_dm(test_dataset=None):
    return BaseDataModule(
        datasets={"train": TinyDataset(), "val": TinyDataset(2), "test": test_dataset},
        collators={"train": ListCollator(), "val": ListCollator(), "test": None},
        dataloader_configs={
            "train": {"batch_size": 2},
            "val": {"batch_size": 2},
            "test": {},
        },
    )


def test_train_dataloader_built():
    dm = make_dm()

    loader = dm.train_dataloader()

    assert isinstance(loader, DataLoader)
    batch = next(iter(loader))
    assert batch["input"] == [0, 1]


def test_none_phase_dataset_yields_none_loader():
    dm = make_dm(test_dataset=None)

    assert dm.test_dataloader() is None


def test_missing_collator_for_active_phase_raises():
    with pytest.raises(KeyError, match="Missing collator for phase 'train'"):
        BaseDataModule(
            datasets={"train": TinyDataset()},
            collators={},
            dataloader_configs={"train": {}},
        )


def test_unknown_phase_raises():
    dm = make_dm()

    with pytest.raises(KeyError, match="Unknown phase 'predict'"):
        dm.make_dataloader("predict")
