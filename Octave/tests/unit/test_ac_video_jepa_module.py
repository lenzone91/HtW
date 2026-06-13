from copy import deepcopy

import pytest
import torch

from Octave.src.Models.Model.ac_video_jepa.configs import DEFAULT_AC_VIDEO_JEPA_MODEL_CONFIG
from Octave.src.Models.Model.ac_video_jepa.factory import build_ac_video_jepa
from Octave.src.Models.Modules.ac_video_jepa_module import AcVideoJepaModule
from Octave.src.Models.Modules.configs import DEFAULT_AC_VIDEO_JEPA_MODULE_CONFIG
from Octave.src.Models.Modules.factory import build_ac_video_jepa_module
from Octave.src.Training.Optimization.configs import DEFAULT_ADAMW_CONFIG


def make_tiny_model_config() -> dict:
    config = deepcopy(DEFAULT_AC_VIDEO_JEPA_MODEL_CONFIG)
    config["encoder"].update(
        {
            "stack_sizes": [4, 8, 8],
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
    config["inverse_dynamics_model"].update(
        {
            "hidden_dim": 16,
            "action_dim": 2,
        }
    )
    config["regularizer"].update(
        {
            "cov_coeff": 0.1,
            "std_coeff": 0.1,
            "sim_coeff_t": 0.1,
            "idm_coeff": 0.1,
        }
    )
    return config


def make_unroll_config() -> dict:
    return {
        "nsteps": 2,
        "unroll_mode": "autoregressive",
        "ctxt_window_time": 1,
        "return_all_steps": False,
    }


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
    model = build_ac_video_jepa(config=make_tiny_model_config())
    optimizer_config = deepcopy(DEFAULT_ADAMW_CONFIG)
    optimizer_config["lr"] = 1e-3

    return AcVideoJepaModule(
        model=model,
        unroll_config=make_unroll_config(),
        optimizer_config=optimizer_config,
        scheduler_config={"enabled": False},
    )


def make_tiny_module_config() -> dict:
    config = deepcopy(DEFAULT_AC_VIDEO_JEPA_MODULE_CONFIG)
    config["model_config"] = make_tiny_model_config()
    config["unroll_config"] = make_unroll_config()
    config["optimizer_config"]["lr"] = 1e-3
    return config


def test_ac_video_jepa_module_compute_step_loss_returns_scalar_and_logs() -> None:
    module = make_module()

    loss, log_dict = module.compute_step_loss(make_batch())

    assert loss.shape == torch.Size([])
    assert log_dict["loss"] is loss
    assert "regularizer/cov_loss" in log_dict
    assert "regularizer/idm_loss" in log_dict


def test_ac_video_jepa_module_training_step_returns_loss() -> None:
    module = make_module()

    loss = module.training_step(make_batch(), batch_idx=0)

    assert loss.shape == torch.Size([])


def test_ac_video_jepa_module_configure_optimizers_returns_optimizer() -> None:
    module = make_module()

    optimizer = module.configure_optimizers()

    assert isinstance(optimizer, torch.optim.AdamW)


def test_ac_video_jepa_module_configure_optimizers_returns_scheduler_dict() -> None:
    module = make_module()
    module.scheduler_config = {
        "enabled": True,
        "scheduler_type": "step_lr",
        "step_size": 1,
        "gamma": 0.5,
        "last_epoch": -1,
        "interval": "epoch",
        "frequency": 1,
        "monitor": None,
        "strict": True,
        "name": None,
    }

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
    assert module.unroll_config["nsteps"] == 2


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
