"""
Tests for the WaveUNet Lightning module wrapper.

This file validates WaveUNet-specific Lightning wrapping while avoiding
Lightning logging internals.
"""

from typing import Any

import torch
from torch import nn

from Project_1_SepFormer_TSE_HF.idea_1_project_setup.src.Modules.tse_base import (
    TSEBaseLightningModule,
)
from Project_1_SepFormer_TSE_HF.idea_1_project_setup.src.Modules.waveUnet import (
    WaveUNetLightningModule,
)


#############################################
# Dummy objects
#############################################


class DummyWaveUNet(nn.Module):
    def forward(self, mixture: torch.Tensor) -> torch.Tensor:
        return mixture


class DummyMetricSet(nn.Module):
    def forward(
        self,
        preds: torch.Tensor,
        target: torch.Tensor,
        mixture: torch.Tensor | None = None,
        clue: Any | None = None,
    ) -> dict[str, torch.Tensor]:
        return {"dummy_metric": torch.tensor(1.0)}


class DummyLoss(nn.Module):
    def forward(
        self,
        metric_logs: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        return torch.tensor(2.0), {"loss": torch.tensor(2.0)}


def make_waveunet_lightning_module() -> WaveUNetLightningModule:
    return WaveUNetLightningModule(
        waveunet=DummyWaveUNet(),
        optimizer_configs={},
        scheduler_configs={},
        train_metrics=DummyMetricSet(),
        val_metrics=DummyMetricSet(),
        test_metrics=DummyMetricSet(),
        loss=DummyLoss(),
        log_loss_ml_steps=("train", "val"),
    )


def disable_lightning_logging(
    module: WaveUNetLightningModule,
) -> None:
    module.log_step_dict = lambda *args, **kwargs: None


#############################################
# Construction
#############################################


def test_waveunet_lightning_module_is_tse_base_lightning_module() -> None:
    module = make_waveunet_lightning_module()

    assert isinstance(module, TSEBaseLightningModule)


def test_waveunet_lightning_module_stores_waveunet_model() -> None:
    module = make_waveunet_lightning_module()

    assert isinstance(module.model, DummyWaveUNet)


#############################################
# Forward
#############################################


def test_waveunet_lightning_module_forward_runs() -> None:
    module = make_waveunet_lightning_module()

    mixture = torch.tensor([[1.0, 2.0, 3.0]])
    preds = module(mixture)

    assert torch.equal(preds, mixture)


#############################################
# Step methods
#############################################


def test_waveunet_lightning_module_training_step_returns_loss() -> None:
    module = make_waveunet_lightning_module()
    disable_lightning_logging(module)

    mixture = torch.tensor([[1.0, 2.0, 3.0]])
    target = torch.tensor([[1.0, 1.0, 1.0]])

    loss = module.training_step(
        batch={
            "mixture": mixture,
            "target": target,
            "clue": None,
            "metadata": {},
        },
        batch_idx=0,
    )

    assert torch.equal(loss, torch.tensor(2.0))


def test_waveunet_lightning_module_validation_step_runs() -> None:
    module = make_waveunet_lightning_module()
    disable_lightning_logging(module)

    mixture = torch.tensor([[1.0, 2.0, 3.0]])
    target = torch.tensor([[1.0, 1.0, 1.0]])

    output = module.validation_step(
        batch={
            "mixture": mixture,
            "target": target,
            "clue": None,
            "metadata": {},
        },
        batch_idx=0,
    )

    assert output is None


def test_waveunet_lightning_module_test_step_runs() -> None:
    module = make_waveunet_lightning_module()
    disable_lightning_logging(module)

    mixture = torch.tensor([[1.0, 2.0, 3.0]])
    target = torch.tensor([[1.0, 1.0, 1.0]])

    output = module.test_step(
        batch={
            "mixture": mixture,
            "target": target,
            "clue": None,
            "metadata": {},
        },
        batch_idx=0,
    )

    assert output is None