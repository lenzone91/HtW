"""
Unit tests for AIML.Data.Collators.factory.
"""

import pytest

from src.AIML.Data.Collators.base import (
    BaseCollator,
)
from src.AIML.Data.Collators.factory import (
    build_collator,
)
from src.AIML.Data.Collators.registry import (
    COLLATOR_REGISTRY,
)
from src.Workflow.Factory.errors import (
    RegistryError,
)


class PassThrough(BaseCollator):
    def __init__(self, tag="t", batch_transforms=None):
        super().__init__(batch_transforms=batch_transforms)
        self.tag = tag

    def collate_samples(self, samples):
        return {"input": [s["input"] for s in samples], "tag": self.tag}


@pytest.fixture
def registered_collator():
    name = "passthrough_collator_under_test"
    COLLATOR_REGISTRY.add_entry(
        name=name,
        object_cls=PassThrough,
        default_config={"tag": "t"},
    )
    yield name
    COLLATOR_REGISTRY.entries.pop(name, None)


def test_build_collator_by_name(registered_collator):
    collator = build_collator({"tag": "x"}, collator_name=registered_collator)

    assert isinstance(collator, PassThrough)
    assert collator([{"input": 1}]) == {"input": [1], "tag": "x"}


def test_build_collator_unknown_name_raises():
    with pytest.raises(RegistryError, match="Unknown collator"):
        build_collator({}, collator_name="missing")
