"""
Integration test for the AIML Data pipeline.

Exercises the full flow with dummy generic objects:

    config dict
      -> build_datamodule (DefaultDataModule)
         -> datasets built per phase (phase_single_named)
         -> collators built per phase, each sub-building its augmentations
      -> DataLoader
      -> batch (collated samples with augmentation applied)
"""

import pytest
import torch
from torch.utils.data import Dataset

from src.AIML.Data.Collators.base import (
    BaseCollator,
)
from src.AIML.Data.Collators.registry import (
    COLLATOR_REGISTRY,
)
from src.AIML.Data.Datasets.base import (
    BaseDataset,
)
from src.AIML.Data.Datasets.registry import (
    DATASET_REGISTRY,
)
from src.AIML.Data.DataAugmentation.base import (
    BaseAugmentation,
)
from src.AIML.Data.DataAugmentation.registry import (
    AUGMENTATION_BUILDER,
    AUGMENTATION_REGISTRY,
)
from src.AIML.Data.DataModules.factory import (
    build_datamodule,
)
from src.Workflow.Factory.registry import (
    SubBuildDeclaration,
)


#############################################
# Dummy generic pipeline objects
#############################################


class RangeDataset(BaseDataset, Dataset):
    def __init__(self, length=4):
        super().__init__()
        self.length = length

    def __len__(self):
        return self.length

    def __getitem__(self, i):
        return self.build_sample(input=float(i), target=float(i))


class StackCollator(BaseCollator):
    def collate_samples(self, samples):
        return {
            "input": torch.tensor([s["input"] for s in samples]),
            "target": torch.tensor([s["target"] for s in samples]),
        }


class AddOffset(BaseAugmentation):
    def __init__(self, offset=0.0):
        self.offset = offset

    def transform(self, batch):
        batch["input"] = batch["input"] + self.offset
        return batch


#############################################
# Registration fixture
#############################################


@pytest.fixture
def registered_pipeline():
    DATASET_REGISTRY.add_entry(
        name="range_ds", object_cls=RangeDataset, default_config={"length": None}
    )
    AUGMENTATION_REGISTRY.add_entry(
        name="add_offset", object_cls=AddOffset, default_config={"offset": 0.0}
    )
    COLLATOR_REGISTRY.add_entry(
        name="stack",
        object_cls=StackCollator,
        default_config={"augmentations": {}},
        sub_builds=(
            SubBuildDeclaration(
                source_key="augmentations",
                target_key="batch_transforms",
                builder=AUGMENTATION_BUILDER,
                build_method="named",
            ),
        ),
    )
    yield
    DATASET_REGISTRY.entries.pop("range_ds", None)
    AUGMENTATION_REGISTRY.entries.pop("add_offset", None)
    COLLATOR_REGISTRY.entries.pop("stack", None)


def _config():
    return {
        "default": {
            "datasets": {
                "train": {"range_ds": {"length": 4}},
                "val": None,
                "test": None,
            },
            "collators": {
                "train": {"stack": {"augmentations": {"add_offset": {"offset": 10.0}}}},
                "val": None,
                "test": None,
            },
            "dataloader_configs": {
                "train": {"batch_size": 2},
                "val": {},
                "test": {},
            },
        }
    }


#############################################
# Tests
#############################################


def test_data_flow_produces_augmented_batches(registered_pipeline):
    datamodule = build_datamodule(_config())

    loader = datamodule.train_dataloader()
    batches = list(loader)

    # 4 samples, batch_size 2 -> 2 batches.
    assert len(batches) == 2

    first = batches[0]
    # inputs 0,1 collated then offset by 10 via the augmentation sub-build.
    assert torch.equal(first["input"], torch.tensor([10.0, 11.0]))
    assert torch.equal(first["target"], torch.tensor([0.0, 1.0]))


def test_collator_received_built_augmentation(registered_pipeline):
    datamodule = build_datamodule(_config())

    collator = datamodule.collators["train"]
    assert len(collator.batch_transforms) == 1
    assert isinstance(collator.batch_transforms[0], AddOffset)
    assert collator.batch_transforms[0].offset == 10.0


def test_empty_phases_are_none(registered_pipeline):
    datamodule = build_datamodule(_config())

    assert datamodule.datasets["val"] is None
    assert datamodule.val_dataloader() is None
    assert datamodule.test_dataloader() is None
