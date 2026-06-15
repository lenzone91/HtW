from copy import deepcopy

import pytest
import torch

from Octave.src.Training.Optimization.configs import DEFAULT_ADAMW_CONFIG
from Octave.src.Training.Optimization.factory import (
    ConfiguredOptimizerBuilder,
    build_optimizer,
    build_optimizer_builder,
)


def test_build_optimizer_builds_adamw_from_plain_config() -> None:
    parameter = torch.nn.Parameter(torch.tensor([1.0]))
    config = deepcopy(DEFAULT_ADAMW_CONFIG)
    config["lr"] = 0.01

    optimizer = build_optimizer(
        parameters=[parameter],
        optimizer_config=config,
    )

    assert isinstance(optimizer, torch.optim.AdamW)
    assert optimizer.param_groups[0]["lr"] == 0.01


def test_build_optimizer_builder_returns_configured_callable() -> None:
    parameter = torch.nn.Parameter(torch.tensor([1.0]))
    config = deepcopy(DEFAULT_ADAMW_CONFIG)
    config["lr"] = 0.01

    optimizer_builder = build_optimizer_builder(optimizer_config=config)
    optimizer = optimizer_builder(parameters=[parameter])

    assert isinstance(optimizer_builder, ConfiguredOptimizerBuilder)
    assert isinstance(optimizer, torch.optim.AdamW)
    assert optimizer.param_groups[0]["lr"] == 0.01


def test_build_optimizer_does_not_mutate_input_config() -> None:
    parameter = torch.nn.Parameter(torch.tensor([1.0]))
    config = deepcopy(DEFAULT_ADAMW_CONFIG)
    original_config = deepcopy(config)

    build_optimizer(parameters=[parameter], optimizer_config=config)

    assert config == original_config


def test_build_optimizer_rejects_unknown_optimizer_type() -> None:
    parameter = torch.nn.Parameter(torch.tensor([1.0]))
    config = {
        **DEFAULT_ADAMW_CONFIG,
        "optimizer_type": "unknown",
    }

    with pytest.raises(RuntimeError, match="Unknown optimizer"):
        build_optimizer(parameters=[parameter], optimizer_config=config)


def test_build_optimizer_rejects_unknown_config_key() -> None:
    parameter = torch.nn.Parameter(torch.tensor([1.0]))
    config = {
        **DEFAULT_ADAMW_CONFIG,
        "unknown": 1,
    }

    with pytest.raises(RuntimeError, match="Invalid config keys"):
        build_optimizer(parameters=[parameter], optimizer_config=config)
