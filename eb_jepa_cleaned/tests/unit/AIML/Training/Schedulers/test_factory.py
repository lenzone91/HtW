"""
Unit tests for AIML.Training.Schedulers.factory.
"""

import pytest
import torch

from src.AIML.Training.Schedulers.factory import (
    build_schedulers,
)
from src.Workflow.Factory.errors import (
    RegistryError,
)


def _optimizer():
    return torch.optim.SGD(torch.nn.Linear(2, 2).parameters(), lr=0.1)


def test_build_schedulers_by_type():
    schedulers = build_schedulers(
        {"model": _optimizer()},
        {"model": {"scheduler_type": "step_lr", "step_size": 5}},
    )

    assert isinstance(schedulers["model"], torch.optim.lr_scheduler.StepLR)


def test_missing_optimizer_raises():
    with pytest.raises(KeyError, match="Missing optimizers"):
        build_schedulers({}, {"model": {"scheduler_type": "step_lr"}})


def test_unknown_scheduler_type_raises():
    with pytest.raises(RegistryError, match="Unknown scheduler"):
        build_schedulers({"model": _optimizer()}, {"model": {"scheduler_type": "nope"}})
