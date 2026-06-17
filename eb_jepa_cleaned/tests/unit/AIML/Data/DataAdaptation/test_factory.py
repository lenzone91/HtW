"""
Unit tests for AIML.Data.DataAdaptation.factory.

Registers dummy generic adaptations onto the shared registry and cleans up.
"""

import pytest

from eb_jepa_cleaned.AIML.Data.BatchTransform.base import (
    BaseBatchTransform,
)
from eb_jepa_cleaned.AIML.Data.DataAdaptation.base import (
    BaseAdaptation,
)
from eb_jepa_cleaned.AIML.Data.DataAdaptation.factory import (
    build_adaptation,
    build_adaptations,
)
from eb_jepa_cleaned.AIML.Data.DataAdaptation.registry import (
    ADAPTATION_REGISTRY,
)
from eb_jepa_cleaned.Workflow.Factory.errors import (
    RegistryError,
)


def test_base_adaptation_is_batch_transform():
    assert issubclass(BaseAdaptation, BaseBatchTransform)


class Rename(BaseAdaptation):
    """Toy adaptation: move 'input' to a model-input key."""

    def __init__(self, target_key="model_input"):
        self.target_key = target_key

    def transform(self, batch):
        out = dict(batch)
        out[self.target_key] = out.pop("input")
        return out


@pytest.fixture
def registered_rename():
    name = "rename_adaptation_under_test"
    ADAPTATION_REGISTRY.add_entry(
        name=name,
        object_cls=Rename,
        default_config={"target_key": "model_input"},
    )
    yield name
    ADAPTATION_REGISTRY.entries.pop(name, None)


def test_build_adaptation_by_name(registered_rename):
    adaptation = build_adaptation(
        {"target_key": "spectrogram"}, adaptation_name=registered_rename
    )

    assert isinstance(adaptation, Rename)
    assert adaptation({"input": 7}) == {"spectrogram": 7}


def test_build_adaptations_returns_ordered_list(registered_rename):
    adaptations = build_adaptations({registered_rename: {"target_key": "x"}})

    assert isinstance(adaptations, list)
    assert len(adaptations) == 1
    assert adaptations[0].target_key == "x"


def test_build_adaptations_empty_and_none_return_empty_list():
    assert build_adaptations(None) == []
    assert build_adaptations({}) == []


def test_build_adaptation_unknown_name_raises():
    with pytest.raises(RegistryError, match="Unknown adaptation"):
        build_adaptation({}, adaptation_name="missing")
