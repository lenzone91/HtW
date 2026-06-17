"""
AcVideoJepaModule: building through the generic build_lightning_module (the full
field-resolution chain: encoder -> shape probe -> predictor/metric-set), one
compute_step, optimizer coverage of metric params, and a one-step trainer.fit.
"""

import copy

import lightning.pytorch as pl
import torch
from torch.utils.data import DataLoader, Dataset

# Importing the subpackage registers the module + its build dependencies.
import src.AcVideoJEPA.Models.Modules  # noqa: F401
from src.AcVideoJEPA.Models.Modules.ac_video_jepa_module import (
    AcVideoJepaModule,
)
from src.AIML.Models.Modules.factory import build_lightning_module

C, T, HW, A, D = 2, 4, 16, 2, 8


def _module_config():
    return {
        "ac_video_jepa": {
            "module_type": "ac_video_jepa",
            "encoder": {
                "input_shape": [C, HW, HW],
                "input_channels": C,
                "stack_sizes": [4, 8],
                "num_blocks": 1,
                "mlp_output_dim": D,
            },
            "predictor": {
                "hidden_size": None,
                "action_dim": A,
                "num_layers": 1,
                "use_encoder_final_ln": True,
            },
            "action_encoder": {"action_encoder_type": "identity"},
            "rollout": {
                "rollout_type": "latent",
                "nsteps": 1,
                "unroll_mode": "autoregressive",
                "ctxt_window_time": 1,
                "return_all_steps": False,
            },
            "metrics": {
                "set_type": "metric",
                "metrics": {
                    "autoregressive_prediction_loss": {
                        "prediction_cost": {
                            "prediction_cost_type": "square_loss_seq",
                            "use_projector": False,
                            "proj": None,
                        }
                    },
                    "hinge_std": {
                        "projector": {
                            "enabled": False,
                            "mlp_spec": None,
                            "hidden_multiplier": 4,
                        },
                        "std_margin": 1.0,
                        "first_t_only": False,
                        "spatial_as_samples": False,
                    },
                },
            },
            "loss": {
                "loss_type": "weighted_metric",
                "metric_weights": {
                    "autoregressive_prediction_loss": 1.0,
                    "hinge_std": 1.0,
                },
            },
            "optimizer_configs": {"optimizer": {"optimizer_type": "adam", "lr": 1e-3}},
            "scheduler_configs": {},
        }
    }


def _batch(batch_size=2):
    return {
        "states": torch.randn(batch_size, C, T, HW, HW),
        "actions": torch.randn(batch_size, A, T),
    }


class ClipDataset(Dataset):
    def __init__(self, n=4):
        self.n = n

    def __len__(self):
        return self.n

    def __getitem__(self, index):
        return {
            "states": torch.randn(C, T, HW, HW),
            "actions": torch.randn(A, T),
        }


#############################################
# Build + step
#############################################


def test_builds_through_factory_with_resolved_subobjects():
    module = build_lightning_module(_module_config())
    assert isinstance(module, AcVideoJepaModule)
    assert set(module.models.keys()) == {"encoder", "predictor", "action_encoder"}
    assert module.encoder_shape["feature_dim"] == D
    # predictor hidden_size defaulted to the encoder feature dim
    assert module.models["predictor"].rnn.hidden_size == D


def test_compute_step_differentiable():
    module = build_lightning_module(_module_config())
    total_loss, metric_values, loss_logs = module.compute_step(_batch())
    assert total_loss.ndim == 0 and total_loss.requires_grad
    assert "autoregressive_prediction_loss" in metric_values
    assert "loss" in loss_logs


def test_check_batch_rejects_missing_keys():
    module = build_lightning_module(_module_config())
    import pytest

    with pytest.raises(KeyError):
        module.compute_step({"states": torch.randn(2, C, T, HW, HW)})


#############################################
# Optimizer covers metric params (inverse dynamics)
#############################################


def test_optimizer_covers_inverse_dynamics_params():
    config = copy.deepcopy(_module_config())
    metrics = config["ac_video_jepa"]["metrics"]["metrics"]
    metrics["inverse_dynamics"] = {
        "projector": {"enabled": False, "mlp_spec": None, "hidden_multiplier": 4},
        "inverse_dynamics_model": {"enabled": True, "hidden_dim": 8, "action_dim": A},
        "after_projection": False,
    }
    config["ac_video_jepa"]["loss"]["metric_weights"]["inverse_dynamics"] = 1.0

    module = build_lightning_module(config)
    # The inverse-dynamics model lives inside the metric set, outside `models`.
    optimizer = module.configure_optimizers()[0]
    optimized = {id(p) for group in optimizer.param_groups for p in group["params"]}
    all_params = {id(p) for p in module.parameters()}
    assert optimized == all_params
    # And there genuinely are params beyond encoder + predictor.
    model_params = {id(p) for p in module.models.parameters()}
    assert all_params - model_params  # metric (IDM) params present


#############################################
# One-step fit
#############################################


def test_one_step_fit_runs():
    torch.manual_seed(0)
    module = build_lightning_module(_module_config())
    loader = DataLoader(ClipDataset(), batch_size=2)
    trainer = pl.Trainer(
        fast_dev_run=True,
        accelerator="cpu",
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
    )
    trainer.fit(module, loader)
