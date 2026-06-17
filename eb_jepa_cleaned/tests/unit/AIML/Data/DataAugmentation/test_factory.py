"""
Unit tests for AIML.Data.DataAugmentation.factory.

Registers dummy generic augmentations onto the shared registry and cleans up.
"""

import pytest

from src.AIML.Data.BatchTransform.base import (
    BaseBatchTransform,
)
from src.AIML.Data.DataAugmentation.base import (
    BaseAugmentation,
)
from src.AIML.Data.DataAugmentation.factory import (
    build_augmentation,
    build_augmentations,
)
from src.AIML.Data.DataAugmentation.registry import (
    AUGMENTATION_REGISTRY,
)
from src.Workflow.Factory.errors import (
    RegistryError,
)


class Tag(BaseAugmentation):
    def __init__(self, tag="t"):
        self.tag = tag

    def transform(self, batch):
        return {**batch, self.tag: True}


def test_base_augmentation_is_batch_transform():
    assert issubclass(BaseAugmentation, BaseBatchTransform)


@pytest.fixture
def registered_tag():
    name = "tag_augmentation_under_test"
    AUGMENTATION_REGISTRY.add_entry(
        name=name,
        object_cls=Tag,
        default_config={"tag": "t"},
    )
    yield name
    AUGMENTATION_REGISTRY.entries.pop(name, None)


def test_build_augmentation_by_name(registered_tag):
    augmentation = build_augmentation({"tag": "x"}, augmentation_name=registered_tag)

    assert isinstance(augmentation, Tag)
    assert augmentation({"a": 1}) == {"a": 1, "x": True}


def test_build_augmentations_returns_ordered_list(registered_tag):
    augmentations = build_augmentations({registered_tag: {"tag": "first"}})

    assert isinstance(augmentations, list)
    assert len(augmentations) == 1
    assert augmentations[0].tag == "first"


def test_build_augmentations_empty_and_none_return_empty_list():
    assert build_augmentations(None) == []
    assert build_augmentations({}) == []


def test_build_augmentation_unknown_name_raises():
    with pytest.raises(RegistryError, match="Unknown augmentation"):
        build_augmentation({}, augmentation_name="missing")
