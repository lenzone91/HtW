"""
Tests for the Lightning module factory.

This file validates factory orchestration with monkeypatched sub-builders.
"""

from copy import deepcopy

import pytest
import torch
from torch import nn

from Project_1_SepFormer_TSE_HF.idea_1_project_setup.src.Modules import factory as module_factory
from Project_1_SepFormer_TSE_HF.idea_1_project_setup.src.Modules.factory import (
    build_lightning_module,
    get_model_loading_config,
)
from Project_1_SepFormer_TSE_HF.idea_1_project_setup.src.Modules.waveUnet import (
    WaveUNetLightningModule,
)
from Project_1_SepFormer_TSE_HF.idea_1_project_setup.src.Modules.configs import (
    DEFAULT_LIGHTNING_MODULE_CONFIG,
)


#############################################
# Dummy objects
#############################################


class DummyModel(nn.Module):
    def forward(self, mixture: torch.Tensor) -> torch.Tensor:
        return mixture


class DummyMetricSet(nn.Module):
    def forward(self, **kwargs) -> dict[str, torch.Tensor]:
        return {"dummy_metric": torch.tensor(1.0)}


class DummyLoss(nn.Module):
    def forward(self, metric_logs: dict) -> tuple[torch.Tensor, dict]:
        return torch.tensor(1.0), {"loss": torch.tensor(1.0)}


#############################################
# Helpers
#############################################


def make_lightning_module_config() -> dict:
    return deepcopy(DEFAULT_LIGHTNING_MODULE_CONFIG)


def patch_factory_dependencies(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        module_factory,
        "build_models",
        lambda model_configs, runtime_context=None: {"waveunet": DummyModel()},
    )

    monkeypatch.setattr(
        module_factory,
        "load_models_if_needed",
        lambda models, loading_config=None, runtime_context=None: models,
    )

    monkeypatch.setattr(
        module_factory,
        "build_metric_set",
        lambda config, runtime_context=None: DummyMetricSet(),
    )

    monkeypatch.setattr(
        module_factory,
        "build_loss",
        lambda config, runtime_context=None: DummyLoss(),
    )


#############################################
# Loading config helper
#############################################


def test_get_model_loading_config_returns_none_without_loading_config() -> None:
    assert get_model_loading_config(None) is None


def test_get_model_loading_config_extracts_model_entry() -> None:
    loading_config = {
        "model": {"load": True},
        "other": {"ignored": True},
    }

    model_loading_config = get_model_loading_config(loading_config)

    assert model_loading_config == {"load": True}


#############################################
# Main factory wrapper
#############################################


def test_build_lightning_module_builds_waveunet_module(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    patch_factory_dependencies(monkeypatch)

    module = build_lightning_module(
        lightning_module_configs=make_lightning_module_config(),
    )

    assert isinstance(module, WaveUNetLightningModule)


def test_build_lightning_module_rejects_unknown_module_name(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    patch_factory_dependencies(monkeypatch)

    config = make_lightning_module_config()
    config["unknown"] = config.pop("waveunet")

    with pytest.raises(RuntimeError):
        build_lightning_module(lightning_module_configs=config)


def test_build_lightning_module_rejects_multiple_module_configs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    patch_factory_dependencies(monkeypatch)

    config = make_lightning_module_config()
    config["second_waveunet"] = deepcopy(config["waveunet"])

    with pytest.raises(RuntimeError):
        build_lightning_module(lightning_module_configs=config)


def test_build_lightning_module_forwards_loading_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed_loading_config = {}

    monkeypatch.setattr(
        module_factory,
        "build_models",
        lambda model_configs, runtime_context=None: {"waveunet": DummyModel()},
    )

    def fake_load_models_if_needed(
        models: dict,
        loading_config: dict | None = None,
        runtime_context: dict | None = None,
    ) -> dict:
        observed_loading_config["value"] = loading_config
        return models

    monkeypatch.setattr(
        module_factory,
        "load_models_if_needed",
        fake_load_models_if_needed,
    )

    monkeypatch.setattr(
        module_factory,
        "build_metric_set",
        lambda config, runtime_context=None: DummyMetricSet(),
    )

    monkeypatch.setattr(
        module_factory,
        "build_loss",
        lambda config, runtime_context=None: DummyLoss(),
    )

    build_lightning_module(
        lightning_module_configs=make_lightning_module_config(),
        loading_config={"model": {"load": True}},
    )

    assert observed_loading_config["value"] == {"load": True}