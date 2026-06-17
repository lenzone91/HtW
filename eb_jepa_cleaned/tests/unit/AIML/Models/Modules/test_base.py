"""
Unit tests for AIML.Models.Modules.base.BaseLightningModule.

Covers log-dict merging/prefixing, validation, and configure_optimizers.
"""

import pytest
import torch
from torch import nn

from src.AIML.Models.Modules.base import (
    BaseLightningModule,
)


class TinyModule(BaseLightningModule):
    def forward(self, x):
        return self.models["m"](x)


def _module(scheduler_configs=None):
    return TinyModule(
        models={"m": nn.Linear(2, 2)},
        optimizer_configs={"m": {"optimizer_type": "adam", "lr": 0.01}},
        scheduler_configs=scheduler_configs if scheduler_configs is not None else {},
    )


def test_prepare_for_log_merges_and_prefixes():
    module = _module()

    out = module._prepare_for_log("train", {"loss": 1.0}, {"sisdr": 2.0})

    assert out == {"train/loss": 1.0, "train/sisdr": 2.0}


def test_duplicate_log_key_raises():
    module = _module()

    with pytest.raises(ValueError, match="Duplicate log key"):
        module._prepare_for_log("train", {"loss": 1.0}, {"loss": 2.0})


def test_invalid_ml_step_raises():
    module = _module()

    with pytest.raises(ValueError, match="Invalid ML step"):
        module._prepare_for_log("predict", {"loss": 1.0})


def test_check_log_dict_rejects_non_dict():
    module = _module()

    with pytest.raises(TypeError, match="log_dict to be a dictionary"):
        module.check_log_dict(["loss"])


def test_configure_optimizers_without_schedulers():
    module = _module()

    optimizers = module.configure_optimizers()

    assert isinstance(optimizers, list)
    assert isinstance(optimizers[0], torch.optim.Adam)


def test_configure_optimizers_with_schedulers():
    module = _module(
        scheduler_configs={"m": {"scheduler_type": "step_lr", "step_size": 1}}
    )

    optimizers, schedulers = module.configure_optimizers()

    assert isinstance(optimizers[0], torch.optim.Adam)
    assert isinstance(schedulers[0], torch.optim.lr_scheduler.StepLR)
