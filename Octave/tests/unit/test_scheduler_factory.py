from copy import deepcopy

import pytest
import torch

from Octave.src.Training.Schedulers.configs import DEFAULT_STEP_LR_CONFIG
from Octave.src.Training.Schedulers.factory import build_scheduler


def make_optimizer() -> torch.optim.Optimizer:
    parameter = torch.nn.Parameter(torch.tensor([1.0]))
    return torch.optim.SGD([parameter], lr=0.1)


def test_build_scheduler_returns_none_when_disabled() -> None:
    scheduler = build_scheduler(
        optimizer=make_optimizer(),
        scheduler_config={"enabled": False},
    )

    assert scheduler is None


def test_build_scheduler_builds_lightning_scheduler_dict() -> None:
    config = deepcopy(DEFAULT_STEP_LR_CONFIG)
    config["step_size"] = 2
    config["gamma"] = 0.5

    scheduler_config = build_scheduler(
        optimizer=make_optimizer(),
        scheduler_config=config,
    )

    assert isinstance(scheduler_config["scheduler"], torch.optim.lr_scheduler.StepLR)
    assert scheduler_config["interval"] == "epoch"
    assert scheduler_config["frequency"] == 1


def test_build_scheduler_does_not_mutate_input_config() -> None:
    config = deepcopy(DEFAULT_STEP_LR_CONFIG)
    original_config = deepcopy(config)

    build_scheduler(
        optimizer=make_optimizer(),
        scheduler_config=config,
    )

    assert config == original_config


def test_build_scheduler_rejects_unknown_scheduler_type() -> None:
    config = {
        **DEFAULT_STEP_LR_CONFIG,
        "scheduler_type": "unknown",
    }

    with pytest.raises(KeyError, match="Unknown scheduler_type"):
        build_scheduler(
            optimizer=make_optimizer(),
            scheduler_config=config,
        )


def test_build_scheduler_rejects_unknown_config_key() -> None:
    config = {
        **DEFAULT_STEP_LR_CONFIG,
        "unknown": 1,
    }

    with pytest.raises(KeyError, match="Unknown scheduler config keys"):
        build_scheduler(
            optimizer=make_optimizer(),
            scheduler_config=config,
        )
