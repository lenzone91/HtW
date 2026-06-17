"""
Unit tests for AIML.Data.Datasets.factory.

The concrete datasets are all audio (Phase 3), so these tests register dummy
generic datasets onto the shared registry and clean up afterwards.
"""

import pytest

from eb_jepa_cleaned.AIML.Data.Datasets.base import (
    BaseDataset,
)
from eb_jepa_cleaned.AIML.Data.Datasets.factory import (
    build_dataset,
    build_datasets,
)
from eb_jepa_cleaned.AIML.Data.Datasets.registry import (
    DATASET_REGISTRY,
)
from eb_jepa_cleaned.Workflow.Factory.errors import (
    RegistryError,
)


class DummyDataset(BaseDataset):
    def __init__(self, length=2):
        super().__init__()
        self.length = length

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        return self.build_sample(input=index, target=index, metadata={"i": index})


@pytest.fixture
def registered_dummy():
    """Register a dummy dataset on the shared registry and remove it after."""
    name = "dummy_dataset_under_test"
    DATASET_REGISTRY.add_entry(
        name=name,
        object_cls=DummyDataset,
        default_config={"length": None},
    )
    yield name
    DATASET_REGISTRY.entries.pop(name, None)


def test_build_dataset_by_name(registered_dummy):
    dataset = build_dataset({"length": 5}, dataset_name=registered_dummy)

    assert isinstance(dataset, DummyDataset)
    assert len(dataset) == 5


def test_build_datasets_named(registered_dummy):
    # Without a routing type field, outer keys are the registered names.
    datasets = build_datasets({registered_dummy: {"length": 3}})

    assert set(datasets) == {registered_dummy}
    assert len(datasets[registered_dummy]) == 3


def test_build_dataset_unknown_name_raises():
    with pytest.raises(RegistryError, match="Unknown dataset"):
        build_dataset({}, dataset_name="missing_dataset")


def test_build_dataset_uses_shared_builder(registered_dummy):
    # Sanity: the factory does not construct a fresh builder per call.
    from eb_jepa_cleaned.AIML.Data.Datasets import factory
    from eb_jepa_cleaned.AIML.Data.Datasets.registry import (
        DATASET_BUILDER,
    )

    assert factory.DATASET_BUILDER is DATASET_BUILDER
