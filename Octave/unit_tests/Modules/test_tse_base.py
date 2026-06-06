"""
Tests for the TSE Lightning module base class.

This file validates project-owned TSE metric/loss orchestration while avoiding
Lightning logging internals.
"""

from typing import Any

import pytest
import torch
from torch import nn

from Project_1_SepFormer_TSE_HF.idea_1_project_setup.src.Modules.tse_base import (
    TSEBaseLightningModule,
)


#############################################
# Dummy objects
#############################################


class DummyMetricSet(nn.Module):
    def __init__(self, name: str) -> None:
        super().__init__()
        self.name = name
        self.last_inputs = None

    def forward(
        self,
        preds: torch.Tensor,
        target: torch.Tensor,
        mixture: torch.Tensor | None = None,
        clue: Any | None = None,
    ) -> dict[str, torch.Tensor]:
        self.last_inputs = {
            "preds": preds,
            "target": target,
            "mixture": mixture,
            "clue": clue,
        }

        return {
            f"{self.name}_metric": torch.tensor(1.0),
        }


class DummyLoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.last_metric_logs = None

    def forward(
        self,
        metric_logs: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        self.last_metric_logs = metric_logs

        return (
            torch.tensor(2.0),
            {"loss": torch.tensor(2.0)},
        )


def make_tse_base_lightning_module(
    log_loss_ml_steps: tuple[str, ...] = ("train", "val"),
) -> TSEBaseLightningModule:
    model = nn.Linear(2, 2)

    return TSEBaseLightningModule(
        models={"model": model},
        optimizer_configs={},
        scheduler_configs={},
        train_metrics=DummyMetricSet("train"),
        val_metrics=DummyMetricSet("val"),
        test_metrics=DummyMetricSet("test"),
        loss=DummyLoss(),
        model_name="model",
        log_loss_ml_steps=log_loss_ml_steps,
    )


def disable_lightning_logging(
    module: TSEBaseLightningModule,
) -> None:
    module.log_step_dict = lambda *args, **kwargs: None


#############################################
# Metric-set selection
#############################################


def test_tse_base_lightning_module_get_tse_metric_set_returns_phase_metric_set() -> None:
    module = make_tse_base_lightning_module()

    assert module.get_tse_metric_set("train") is module.train_metrics
    assert module.get_tse_metric_set("val") is module.val_metrics
    assert module.get_tse_metric_set("test") is module.test_metrics


def test_tse_base_lightning_module_get_tse_metric_set_rejects_invalid_step() -> None:
    module = make_tse_base_lightning_module()

    with pytest.raises(ValueError):
        module.get_tse_metric_set("predict")


#############################################
# Metric/loss computation
#############################################


def test_tse_base_lightning_module_compute_tse_metrics_calls_selected_metric_set() -> None:
    module = make_tse_base_lightning_module()

    preds = torch.tensor([[1.0, 2.0]])
    target = torch.tensor([[1.0, 1.0]])
    mixture = torch.tensor([[0.5, 0.5]])

    metric_logs = module.compute_tse_metrics(
        ml_step="train",
        preds=preds,
        target=target,
        mixture=mixture,
        clue="speaker_0",
    )

    assert set(metric_logs.keys()) == {"train_metric"}
    assert module.train_metrics.last_inputs == {
        "preds": preds,
        "target": target,
        "mixture": mixture,
        "clue": "speaker_0",
    }


def test_tse_base_lightning_module_compute_tse_loss_calls_loss() -> None:
    module = make_tse_base_lightning_module()

    metric_logs = {"train_metric": torch.tensor(1.0)}

    loss, loss_logs = module.compute_tse_loss(metric_logs)

    assert torch.equal(loss, torch.tensor(2.0))
    assert loss_logs == {"loss": torch.tensor(2.0)}
    assert module.loss.last_metric_logs is metric_logs


#############################################
# Step processing
#############################################


def test_tse_base_lightning_module_process_train_step_returns_loss() -> None:
    module = make_tse_base_lightning_module()
    disable_lightning_logging(module)

    loss = module.process_tse_step_outputs(
        ml_step="train",
        preds=torch.tensor([[1.0, 2.0]]),
        target=torch.tensor([[1.0, 1.0]]),
        mixture=torch.tensor([[0.5, 0.5]]),
    )

    assert torch.equal(loss, torch.tensor(2.0))


def test_tse_base_lightning_module_process_val_step_returns_loss_if_configured() -> None:
    module = make_tse_base_lightning_module(
        log_loss_ml_steps=("train", "val"),
    )
    disable_lightning_logging(module)

    loss = module.process_tse_step_outputs(
        ml_step="val",
        preds=torch.tensor([[1.0, 2.0]]),
        target=torch.tensor([[1.0, 1.0]]),
        mixture=torch.tensor([[0.5, 0.5]]),
    )

    assert torch.equal(loss, torch.tensor(2.0))


def test_tse_base_lightning_module_process_test_step_returns_none_by_default() -> None:
    module = make_tse_base_lightning_module()
    disable_lightning_logging(module)

    loss = module.process_tse_step_outputs(
        ml_step="test",
        preds=torch.tensor([[1.0, 2.0]]),
        target=torch.tensor([[1.0, 1.0]]),
        mixture=torch.tensor([[0.5, 0.5]]),
    )

    assert loss is None