"""
Unit tests for AIML.Training.Optimizers.factory.
"""

import pytest
import torch

from src.AIML.Training.Optimizers.factory import (
    build_optimizers,
    build_optimizers_from_models,
)
from src.Workflow.Factory.errors import (
    RegistryError,
)


def test_build_optimizers_by_type():
    model = torch.nn.Linear(2, 2)

    optimizers = build_optimizers(
        {"model": model.parameters()},
        {"model": {"optimizer_type": "adam", "lr": 0.01}},
    )

    assert isinstance(optimizers["model"], torch.optim.Adam)


def test_build_optimizers_from_models():
    model = torch.nn.Linear(2, 2)

    optimizers = build_optimizers_from_models(
        {"model": model}, {"model": {"optimizer_type": "sgd"}}
    )

    assert isinstance(optimizers["model"], torch.optim.SGD)


def test_missing_parameter_group_raises():
    with pytest.raises(KeyError, match="Missing parameter groups"):
        build_optimizers({}, {"model": {"optimizer_type": "adam"}})


def test_unknown_optimizer_type_raises():
    model = torch.nn.Linear(2, 2)

    with pytest.raises(RegistryError, match="Unknown optimizer"):
        build_optimizers(
            {"model": model.parameters()}, {"model": {"optimizer_type": "nope"}}
        )
