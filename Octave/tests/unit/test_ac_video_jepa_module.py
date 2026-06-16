from copy import deepcopy

import pytest
import torch

from Octave.src.Metrics.factory import build_ac_video_jepa_metric_stack
from Octave.src.Metrics.Loss.configs import DEFAULT_WEIGHTED_METRIC_LOSS_CONFIG
from Octave.src.Metrics.MetricSets.configs import (
    DEFAULT_AC_VIDEO_JEPA_METRIC_SET_CONFIG,
)
from Octave.src.Models.Model.ac_video_jepa.configs import (
    DEFAULT_AC_VIDEO_JEPA_COMPONENTS_CONFIG,
)
from Octave.src.Models.Model.ac_video_jepa.factory import (
    build_ac_video_jepa_components,
)
from Octave.src.Models.Modules.ac_video_jepa_module import AcVideoJepaModule
from Octave.src.Models.Modules.configs import DEFAULT_AC_VIDEO_JEPA_MODULE_CONFIG
from Octave.src.Models.Modules.factory import build_ac_video_jepa_module
from Octave.src.Rollouts.configs import DEFAULT_LATENT_ROLLOUT_CONFIG
from Octave.src.Rollouts.factory import build_latent_rollout
from Octave.src.Training.Optimization.configs import DEFAULT_ADAMW_CONFIG
from Octave.src.Training.Optimization.factory import build_optimizer_builder
from Octave.src.Training.Schedulers.factory import build_scheduler_builder


def make_tiny_components_config() -> dict:
    config = deepcopy(DEFAULT_AC_VIDEO_JEPA_COMPONENTS_CONFIG)
    config["encoder"].update(
        {
            "stack_sizes": [4, 8, 8],
            "input_channels": 2,
            "input_shape": [2, 32, 32],
            "mlp_output_dim": 32,
        }
    )
    config["predictor"].update(
        {
            "hidden_size": 32,
            "action_dim": 2,
        }
    )
    return config


def make_tiny_metric_set_config() -> dict:
    config = deepcopy(DEFAULT_AC_VIDEO_JEPA_METRIC_SET_CONFIG)
    config["metrics"]["idm_loss"]["inverse_dynamics_model"].update(
        {
            "hidden_dim": 16,
            "action_dim": 2,
        }
    )
    return config


def make_tiny_loss_config() -> dict:
    config = deepcopy(DEFAULT_WEIGHTED_METRIC_LOSS_CONFIG)
    config["metric_weights"].update(
        {
            "prediction_loss": 1.0,
            "cov_loss": 0.1,
            "std_loss": 0.1,
            "sim_loss_t": 0.1,
            "idm_loss": 0.1,
        }
    )
    return config


def make_rollout_config() -> dict:
    config = deepcopy(DEFAULT_LATENT_ROLLOUT_CONFIG)
    config.update(
        {
            "rollout_type": "latent",
            "nsteps": 2,
            "unroll_mode": "autoregressive",
            "ctxt_window_time": 1,
            "return_all_steps": False,
        }
    )
    return config


def make_batch() -> dict:
    return {
        "states": torch.randn(2, 2, 4, 32, 32),
        "actions": torch.randn(2, 2, 4),
        "locations": torch.randn(2, 2, 4),
        "wall_x": torch.tensor([16, 16]),
        "door_y": torch.tensor([10, 10]),
        "metadata": [{"index": 0}, {"index": 1}],
    }


def make_module() -> AcVideoJepaModule:
    components = build_ac_video_jepa_components(config=make_tiny_components_config())
    rollout = build_latent_rollout(config=make_rollout_config())
    metric_stack = build_ac_video_jepa_metric_stack(
        metric_set_config=make_tiny_metric_set_config(),
        loss_config=make_tiny_loss_config(),
        encoder_shape=components["encoder_shape"],
    )
    optimizer_config = deepcopy(DEFAULT_ADAMW_CONFIG)
    optimizer_config["lr"] = 1e-3
    optimizer_builder = build_optimizer_builder(optimizer_config=optimizer_config)
    scheduler_builder = build_scheduler_builder(scheduler_config={"enabled": False})

    return AcVideoJepaModule(
        encoder=components["encoder"],
        action_encoder=components["action_encoder"],
        predictor=components["predictor"],
        encoder_shape=components["encoder_shape"],
        rollout=rollout,
        metric_set=metric_stack["metric_set"],
        loss=metric_stack["loss"],
        optimizer_builder=optimizer_builder,
        scheduler_builder=scheduler_builder,
    )


def make_tiny_module_config() -> dict:
    config = deepcopy(DEFAULT_AC_VIDEO_JEPA_MODULE_CONFIG)
    config["components_config"] = make_tiny_components_config()
    config["rollout_config"] = make_rollout_config()
    config["metric_set_config"] = make_tiny_metric_set_config()
    config["loss_config"] = make_tiny_loss_config()
    config["optimizer_config"]["lr"] = 1e-3
    return config


def test_ac_video_jepa_module_compute_step_loss_returns_scalar_and_logs() -> None:
    module = make_module()

    loss, log_dict = module.compute_step_loss(make_batch())

    assert loss.shape == torch.Size([])
    assert log_dict["loss"] is loss
    assert "prediction_loss" in log_dict
    assert "cov_loss" in log_dict
    assert "idm_loss" in log_dict
    assert "loss/cov_loss" in log_dict


def test_ac_video_jepa_module_forward_returns_loss_free_rollout() -> None:
    module = make_module()
    batch = make_batch()

    rollout_output = module(batch["states"], batch["actions"])

    assert rollout_output.predicted_states.shape == torch.Size([2, 32, 3, 1, 1])
    assert not hasattr(rollout_output, "losses")


def test_ac_video_jepa_module_training_step_returns_loss() -> None:
    module = make_module()

    loss = module.training_step(make_batch(), batch_idx=0)

    assert loss.shape == torch.Size([])


def test_ac_video_jepa_module_training_step_logs_to_logger_on_step_and_epoch(
    monkeypatch,
) -> None:
    module = make_module()
    log_calls = []

    monkeypatch.setattr(
        module,
        "log_step_dict",
        lambda *args, **kwargs: log_calls.append((args, kwargs)),
    )

    module.training_step(make_batch(), batch_idx=0)

    assert len(log_calls) == 1
    args, kwargs = log_calls[0]
    assert args[0] == "train"
    assert kwargs == {
        "prog_bar": True,
        "on_step": True,
        "on_epoch": True,
        "logger": True,
    }


def test_ac_video_jepa_module_validation_step_logs_to_logger_on_epoch(
    monkeypatch,
) -> None:
    module = make_module()
    log_calls = []

    monkeypatch.setattr(
        module,
        "log_step_dict",
        lambda *args, **kwargs: log_calls.append((args, kwargs)),
    )

    module.validation_step(make_batch(), batch_idx=0)

    assert len(log_calls) == 1
    args, kwargs = log_calls[0]
    assert args[0] == "val"
    assert kwargs == {
        "prog_bar": False,
        "on_step": False,
        "on_epoch": True,
        "logger": True,
    }


def test_ac_video_jepa_module_configure_optimizers_returns_optimizer() -> None:
    module = make_module()

    optimizer = module.configure_optimizers()

    assert isinstance(optimizer, torch.optim.AdamW)


def test_ac_video_jepa_module_configure_optimizers_returns_scheduler_dict() -> None:
    module = make_module()
    module.scheduler_builder = build_scheduler_builder(scheduler_config={
        "enabled": True,
        "scheduler_type": "step_lr",
        "step_size": 1,
        "gamma": 0.5,
        "last_epoch": -1,
        "interval": "epoch",
        "frequency": 1,
        "monitor": None,
        "name": None,
    })

    optimizer_config = module.configure_optimizers()

    assert isinstance(optimizer_config["optimizer"], torch.optim.AdamW)
    assert isinstance(
        optimizer_config["lr_scheduler"]["scheduler"],
        torch.optim.lr_scheduler.StepLR,
    )


def test_ac_video_jepa_module_rejects_missing_batch_key() -> None:
    module = make_module()
    batch = make_batch()
    batch.pop("actions")

    with pytest.raises(KeyError, match="missing required keys"):
        module.compute_step_loss(batch)


def test_ac_video_jepa_module_rejects_non_tensor_batch_value() -> None:
    module = make_module()
    batch = make_batch()
    batch["states"] = "not tensor"

    with pytest.raises(TypeError, match="must be a torch.Tensor"):
        module.compute_step_loss(batch)


def test_build_ac_video_jepa_module_builds_module_from_plain_config() -> None:
    module = build_ac_video_jepa_module(config=make_tiny_module_config())

    assert isinstance(module, AcVideoJepaModule)
    assert module.rollout.nsteps == 2
    assert module.encoder_shape["feature_dim"] == 32


def test_build_ac_video_jepa_module_does_not_mutate_input_config() -> None:
    config = make_tiny_module_config()
    original_config = deepcopy(config)

    build_ac_video_jepa_module(config=config)

    assert config == original_config


def test_build_ac_video_jepa_module_rejects_unknown_key() -> None:
    config = {
        **make_tiny_module_config(),
        "unknown": {},
    }

    with pytest.raises(KeyError, match="Unknown AcVideoJepa module config keys"):
        build_ac_video_jepa_module(config=config)
