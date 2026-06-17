"""
Unit tests for AIML.Data.DataModules.factory.

Builds the generic DefaultDataModule from a config dict, with dummy datasets and
collators registered on the shared registries.
"""

import pytest

from eb_jepa_cleaned.AIML.Data.Collators.base import (
    BaseCollator,
)
from eb_jepa_cleaned.AIML.Data.Collators.registry import (
    COLLATOR_REGISTRY,
)
from eb_jepa_cleaned.AIML.Data.Datasets.base import (
    BaseDataset,
)
from eb_jepa_cleaned.AIML.Data.Datasets.registry import (
    DATASET_REGISTRY,
)
from eb_jepa_cleaned.AIML.Data.DataModules.default import (
    DEFAULT_DATAMODULE_CONFIG,
    DefaultDataModule,
)
from eb_jepa_cleaned.AIML.Data.DataModules.factory import (
    build_datamodule,
)
from eb_jepa_cleaned.AIML.Data.DataModules.registry import (
    DATAMODULE_REGISTRY,
    DATAMODULE_SUB_BUILDS,
)


class TinyDataset(BaseDataset):
    def __init__(self, length=4):
        super().__init__()
        self.length = length

    def __len__(self):
        return self.length

    def __getitem__(self, i):
        return self.build_sample(input=i, target=i)


class ListCollator(BaseCollator):
    def collate_samples(self, samples):
        return {"input": [s["input"] for s in samples]}


@pytest.fixture
def registered(request):
    DATASET_REGISTRY.add_entry(
        name="tiny_ds", object_cls=TinyDataset, default_config={"length": None}
    )
    COLLATOR_REGISTRY.add_entry(
        name="list_collator", object_cls=ListCollator, default_config={}
    )
    yield
    DATASET_REGISTRY.entries.pop("tiny_ds", None)
    COLLATOR_REGISTRY.entries.pop("list_collator", None)


def _config():
    return {
        "default": {
            "datasets": {
                "train": {"tiny_ds": {"length": 4}},
                "val": {"tiny_ds": {"length": 2}},
                "test": None,
            },
            "collators": {
                "train": {"list_collator": {}},
                "val": {"list_collator": {}},
                "test": None,
            },
            "dataloader_configs": {
                "train": {"batch_size": 2},
                "val": {"batch_size": 2},
                "test": {},
            },
        }
    }


def test_build_datamodule_builds_default(registered):
    dm = build_datamodule(_config())

    assert isinstance(dm, DefaultDataModule)


def test_phase_objects_built_and_none_preserved(registered):
    dm = build_datamodule(_config())

    assert isinstance(dm.datasets["train"], TinyDataset)
    assert len(dm.datasets["train"]) == 4
    assert isinstance(dm.collators["val"], ListCollator)
    assert dm.datasets["test"] is None
    assert dm.collators["test"] is None


def test_build_datamodule_rejects_multiple(registered):
    # Register a second valid datamodule so build_named succeeds for both keys
    # and the "exactly one" guard is what rejects the config.
    DATAMODULE_REGISTRY.add_entry(
        name="second",
        object_cls=DefaultDataModule,
        default_config=DEFAULT_DATAMODULE_CONFIG,
        sub_builds=DATAMODULE_SUB_BUILDS,
    )
    try:
        config = _config()
        config["second"] = _config()["default"]

        with pytest.raises(ValueError, match="exactly one named DataModule"):
            build_datamodule(config)
    finally:
        DATAMODULE_REGISTRY.entries.pop("second", None)
