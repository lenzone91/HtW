"""
Unit tests for AIML.Models.Models.factory.
"""

import pytest
import torch
from torch import nn

from eb_jepa_cleaned.AIML.Models.Models.factory import (
    build_model,
    build_models,
)
from eb_jepa_cleaned.AIML.Models.Models.registry import (
    MODEL_REGISTRY,
)
from eb_jepa_cleaned.Workflow.Factory.errors import (
    RegistryError,
)


class Tiny(nn.Module):
    def __init__(self, in_dim=2, out_dim=2):
        super().__init__()
        self.lin = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        return self.lin(x)


@pytest.fixture
def registered_model():
    name = "tiny_model_under_test"
    MODEL_REGISTRY.add_entry(
        name=name, object_cls=Tiny, default_config={"in_dim": 2, "out_dim": 2}
    )
    yield name
    MODEL_REGISTRY.entries.pop(name, None)


def test_build_model_by_name(registered_model):
    model = build_model({"in_dim": 4, "out_dim": 3}, model_name=registered_model)

    assert isinstance(model, Tiny)
    assert model(torch.zeros(1, 4)).shape == (1, 3)


def test_build_models_named(registered_model):
    models = build_models({registered_model: {"in_dim": 2, "out_dim": 2}})

    assert set(models) == {registered_model}
    assert isinstance(models[registered_model], Tiny)


def test_build_model_unknown_name_raises():
    with pytest.raises(RegistryError, match="Unknown model"):
        build_model({}, model_name="missing")
