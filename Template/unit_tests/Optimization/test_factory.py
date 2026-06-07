"""
Pytest test module for Optimization.factory.

Pytest automatically discovers and runs functions whose names start with
``test_``. Therefore, no explicit test decorator is needed here.
"""

import pytest
import torch
from torch import nn

from Project_1_SepFormer_TSE_HF.idea_1_project_setup.src.Optimization.configs import (
    DEFAULT_ADAM_CONFIG,
    DEFAULT_ADAMW_CONFIG,
    DEFAULT_COSINE_ANNEALING_CONFIG,
    DEFAULT_EXPONENTIAL_LR_CONFIG,
    DEFAULT_MULTISTEP_LR_CONFIG,
    DEFAULT_OPTIMIZER_CONFIGS,
    DEFAULT_REDUCE_ON_PLATEAU_CONFIG,
    DEFAULT_SCHEDULER_CONFIGS,
    DEFAULT_SGD_CONFIG,
    DEFAULT_STEP_LR_CONFIG,
)
from Project_1_SepFormer_TSE_HF.idea_1_project_setup.src.Optimization.factory import (
    build_optimizers,
    build_optimizers_from_models,
    build_schedulers,
)


#############################################
# Test data
#############################################

class TinyModel(nn.Module):
    """
    Minimal trainable model used to provide parameters.
    """

    def __init__(self) -> None:
        super().__init__()

        self.linear = nn.Linear(2, 1)


def make_model() -> TinyModel:
    """
    Build a fresh model for optimizer tests.
    """
    return TinyModel()


def make_parameter_groups() -> dict:
    """
    Build fresh parameter groups for optimizer tests.
    """
    return {
        "model": make_model().parameters(),
    }


def make_optimizer() -> torch.optim.Optimizer:
    """
    Build a fresh optimizer for scheduler tests.
    """
    return torch.optim.AdamW(
        make_model().parameters(),
        lr=1e-3,
    )


#############################################
# Optimizer tests
#############################################

def test_build_optimizers_builds_default_optimizer() -> None:
    """
    Test that default optimizer configs build successfully.
    """
    optimizers = build_optimizers(
        parameter_groups=make_parameter_groups(),
        optimizer_configs=DEFAULT_OPTIMIZER_CONFIGS,
        runtime_context=None,
    )

    assert isinstance(optimizers, dict)
    assert isinstance(optimizers["model"], torch.optim.AdamW)


def test_build_optimizers_from_models_extracts_parameters() -> None:
    """
    Test that optimizers can be built directly from model dictionaries.
    """
    optimizers = build_optimizers_from_models(
        models={
            "model": make_model(),
        },
        optimizer_configs=DEFAULT_OPTIMIZER_CONFIGS,
        runtime_context=None,
    )

    assert isinstance(optimizers["model"], torch.optim.AdamW)


def test_build_optimizers_builds_sgd() -> None:
    """
    Test SGD optimizer build.
    """
    config = {
        "model": {
            "optimizer_type": "sgd",
            **DEFAULT_SGD_CONFIG,
        },
    }

    optimizers = build_optimizers(
        parameter_groups=make_parameter_groups(),
        optimizer_configs=config,
        runtime_context=None,
    )

    assert isinstance(optimizers["model"], torch.optim.SGD)


def test_build_optimizers_builds_adam() -> None:
    """
    Test Adam optimizer build.
    """
    config = {
        "model": {
            "optimizer_type": "adam",
            **DEFAULT_ADAM_CONFIG,
        },
    }

    optimizers = build_optimizers(
        parameter_groups=make_parameter_groups(),
        optimizer_configs=config,
        runtime_context=None,
    )

    assert isinstance(optimizers["model"], torch.optim.Adam)


def test_build_optimizers_builds_adamw() -> None:
    """
    Test AdamW optimizer build.
    """
    config = {
        "model": {
            "optimizer_type": "adamw",
            **DEFAULT_ADAMW_CONFIG,
        },
    }

    optimizers = build_optimizers(
        parameter_groups=make_parameter_groups(),
        optimizer_configs=config,
        runtime_context=None,
    )

    assert isinstance(optimizers["model"], torch.optim.AdamW)


def test_build_optimizers_rejects_unknown_optimizer_type() -> None:
    """
    Test that unknown optimizer types fail explicitly.
    """
    config = {
        "model": {
            "optimizer_type": "unknown",
        },
    }

    with pytest.raises(RuntimeError):
        build_optimizers(
            parameter_groups=make_parameter_groups(),
            optimizer_configs=config,
            runtime_context=None,
            strict=True,
        )


def test_build_optimizers_rejects_missing_parameter_group() -> None:
    """
    Test that each optimizer config must have a matching parameter group.
    """
    config = {
        "missing_model": {
            "optimizer_type": "adamw",
            **DEFAULT_ADAMW_CONFIG,
        },
    }

    with pytest.raises(RuntimeError):
        build_optimizers(
            parameter_groups=make_parameter_groups(),
            optimizer_configs=config,
            runtime_context=None,
            strict=True,
        )


#############################################
# Scheduler tests
#############################################

def test_build_schedulers_builds_default_scheduler() -> None:
    """
    Test that default scheduler configs build successfully.
    """
    schedulers = build_schedulers(
        optimizer_groups={
            "model": make_optimizer(),
        },
        scheduler_configs=DEFAULT_SCHEDULER_CONFIGS,
        runtime_context=None,
    )

    assert isinstance(schedulers, dict)
    assert isinstance(schedulers["model"], torch.optim.lr_scheduler.StepLR)


def test_build_schedulers_builds_step_lr() -> None:
    """
    Test StepLR scheduler build.
    """
    config = {
        "model": {
            "scheduler_type": "step_lr",
            **DEFAULT_STEP_LR_CONFIG,
        },
    }

    schedulers = build_schedulers(
        optimizer_groups={
            "model": make_optimizer(),
        },
        scheduler_configs=config,
        runtime_context=None,
    )

    assert isinstance(schedulers["model"], torch.optim.lr_scheduler.StepLR)


def test_build_schedulers_builds_multistep_lr() -> None:
    """
    Test MultiStepLR scheduler build.
    """
    config = {
        "model": {
            "scheduler_type": "multistep_lr",
            **DEFAULT_MULTISTEP_LR_CONFIG,
        },
    }

    schedulers = build_schedulers(
        optimizer_groups={
            "model": make_optimizer(),
        },
        scheduler_configs=config,
        runtime_context=None,
    )

    assert isinstance(schedulers["model"], torch.optim.lr_scheduler.MultiStepLR)


def test_build_schedulers_builds_exponential_lr() -> None:
    """
    Test ExponentialLR scheduler build.
    """
    config = {
        "model": {
            "scheduler_type": "exponential_lr",
            **DEFAULT_EXPONENTIAL_LR_CONFIG,
        },
    }

    schedulers = build_schedulers(
        optimizer_groups={
            "model": make_optimizer(),
        },
        scheduler_configs=config,
        runtime_context=None,
    )

    assert isinstance(schedulers["model"], torch.optim.lr_scheduler.ExponentialLR)


def test_build_schedulers_builds_cosine_annealing() -> None:
    """
    Test CosineAnnealingLR scheduler build.
    """
    config = {
        "model": {
            "scheduler_type": "cosine_annealing",
            **DEFAULT_COSINE_ANNEALING_CONFIG,
        },
    }

    schedulers = build_schedulers(
        optimizer_groups={
            "model": make_optimizer(),
        },
        scheduler_configs=config,
        runtime_context=None,
    )

    assert isinstance(schedulers["model"], torch.optim.lr_scheduler.CosineAnnealingLR)


def test_build_schedulers_builds_reduce_on_plateau() -> None:
    """
    Test ReduceLROnPlateau scheduler build.
    """
    config = {
        "model": {
            "scheduler_type": "reduce_on_plateau",
            **DEFAULT_REDUCE_ON_PLATEAU_CONFIG,
        },
    }

    schedulers = build_schedulers(
        optimizer_groups={
            "model": make_optimizer(),
        },
        scheduler_configs=config,
        runtime_context=None,
    )

    assert isinstance(
        schedulers["model"],
        torch.optim.lr_scheduler.ReduceLROnPlateau,
    )


def test_build_schedulers_rejects_unknown_scheduler_type() -> None:
    """
    Test that unknown scheduler types fail explicitly.
    """
    config = {
        "model": {
            "scheduler_type": "unknown",
        },
    }

    with pytest.raises(RuntimeError):
        build_schedulers(
            optimizer_groups={
                "model": make_optimizer(),
            },
            scheduler_configs=config,
            runtime_context=None,
            strict=True,
        )


def test_build_schedulers_rejects_missing_optimizer_group() -> None:
    """
    Test that each scheduler config must have a matching optimizer.
    """
    config = {
        "missing_model": {
            "scheduler_type": "step_lr",
            **DEFAULT_STEP_LR_CONFIG,
        },
    }

    with pytest.raises(RuntimeError):
        build_schedulers(
            optimizer_groups={
                "model": make_optimizer(),
            },
            scheduler_configs=config,
            runtime_context=None,
            strict=True,
        )