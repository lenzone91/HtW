"""
Tests for the generic Lightning module base class.

This file validates project-owned logging helpers and optimizer orchestration.
"""

import pytest
import torch
from torch import nn
from torch.optim import Optimizer

from Project_1_SepFormer_TSE_HF.idea_1_project_setup.src.Modules.base import (
    BaseLightningModule,
)


#############################################
# Helpers
#############################################


def make_base_lightning_module(
    scheduler_configs: dict | None = None,
) -> BaseLightningModule:
    model = nn.Linear(2, 1)

    optimizer_configs = {
        "model": {
            "optimizer_type": "adam",
            "lr": 1e-3,
            "weight_decay": 0.0,
        },
    }

    if scheduler_configs is None:
        scheduler_configs = {}

    return BaseLightningModule(
        models={"model": model},
        optimizer_configs=optimizer_configs,
        scheduler_configs=scheduler_configs,
    )


#############################################
# Initialization
#############################################


def test_base_lightning_module_stores_models_as_module_dict() -> None:
    module = make_base_lightning_module()

    assert isinstance(module.models, nn.ModuleDict)
    assert isinstance(module.models["model"], nn.Linear)


#############################################
# ML step checks
#############################################


def test_base_lightning_module_check_ml_step_accepts_valid_steps() -> None:
    module = make_base_lightning_module()

    module.check_ml_step("train")
    module.check_ml_step("val")
    module.check_ml_step("test")


def test_base_lightning_module_check_ml_step_rejects_invalid_step() -> None:
    module = make_base_lightning_module()

    with pytest.raises(ValueError):
        module.check_ml_step("predict")


#############################################
# Log dict checks
#############################################


def test_base_lightning_module_check_log_dict_accepts_string_keys() -> None:
    module = make_base_lightning_module()

    module.check_log_dict({"loss": torch.tensor(1.0)})


def test_base_lightning_module_check_log_dict_rejects_non_dict() -> None:
    module = make_base_lightning_module()

    with pytest.raises(TypeError):
        module.check_log_dict([("loss", torch.tensor(1.0))])


def test_base_lightning_module_check_log_dict_rejects_non_string_key() -> None:
    module = make_base_lightning_module()

    with pytest.raises(TypeError):
        module.check_log_dict({0: torch.tensor(1.0)})


#############################################
# Log dict manipulation
#############################################


def test_base_lightning_module_merge_log_dicts_merges_distinct_keys() -> None:
    module = make_base_lightning_module()

    merged_log_dict = module._merge_log_dicts(
        {"loss": torch.tensor(1.0)},
        {"sisdr": torch.tensor(2.0)},
    )

    assert set(merged_log_dict.keys()) == {"loss", "sisdr"}
    assert torch.equal(merged_log_dict["loss"], torch.tensor(1.0))
    assert torch.equal(merged_log_dict["sisdr"], torch.tensor(2.0))


def test_base_lightning_module_merge_log_dicts_rejects_duplicate_keys() -> None:
    module = make_base_lightning_module()

    with pytest.raises(ValueError):
        module._merge_log_dicts(
            {"loss": torch.tensor(1.0)},
            {"loss": torch.tensor(2.0)},
        )


def test_base_lightning_module_add_log_prefix() -> None:
    module = make_base_lightning_module()

    prefixed_log_dict = module._add_log_prefix(
        log_dict={"loss": torch.tensor(1.0)},
        prefix="train/",
    )

    assert set(prefixed_log_dict.keys()) == {"train/loss"}
    assert torch.equal(prefixed_log_dict["train/loss"], torch.tensor(1.0))


def test_base_lightning_module_prepare_for_log_merges_and_prefixes() -> None:
    module = make_base_lightning_module()

    log_dict = module._prepare_for_log(
        "train",
        {"loss": torch.tensor(1.0)},
        {"sisdr": torch.tensor(2.0)},
    )

    assert set(log_dict.keys()) == {"train/loss", "train/sisdr"}
    assert torch.equal(log_dict["train/loss"], torch.tensor(1.0))
    assert torch.equal(log_dict["train/sisdr"], torch.tensor(2.0))


#############################################
# Optimizer configuration
#############################################


def test_base_lightning_module_configure_optimizers_builds_optimizers() -> None:
    module = make_base_lightning_module()

    optimizers = module.configure_optimizers()

    assert isinstance(optimizers, list)
    assert len(optimizers) == 1
    assert isinstance(optimizers[0], Optimizer)


def test_base_lightning_module_configure_optimizers_returns_only_optimizers_without_schedulers() -> None:
    module = make_base_lightning_module()

    optimizers = module.configure_optimizers()

    assert isinstance(optimizers, list)
    assert all(isinstance(optimizer, Optimizer) for optimizer in optimizers)