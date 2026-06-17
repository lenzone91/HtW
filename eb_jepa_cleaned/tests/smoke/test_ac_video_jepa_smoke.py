"""
End-to-end smoke test for AcVideoJEPA.

Composes a full plain-dict run config, builds it through the AIML Execution
composition factory (datamodule + module + Trainer), and runs a one-step
trainer.fit on real (tiny) two-rooms data. This exercises the entire pipeline:
dataset -> collator -> DataLoader -> encode -> latent rollout -> JEPA objective
metrics -> weighted loss -> optimizer step.

Hydra config composition and Workflow/Setup runtime_context are not exercised
here (the config is authored inline and runtime_context is None); they are a
separate step.
"""

import torch

# Register the experiment concretes (module + data) onto the AIML registries.
import src.AcVideoJEPA.Data.Collators  # noqa: F401
import src.AcVideoJEPA.Data.Datasets  # noqa: F401
import src.AcVideoJEPA.Models.Modules  # noqa: F401
from src.AIML.Execution.Runs.factory import build_training_objects

IMG, C, A, SAMPLE_LEN, D = 16, 2, 2, 3, 32

DATASET_CONFIG = {
    "size": 4,
    "val_size": 2,
    "n_steps": 6,
    "sample_length": SAMPLE_LEN,
    "img_size": IMG,
    "batch_size": 2,
    "train": True,
    "device": "cpu",
}

RUN_CONFIG = {
    "datamodule": {
        "default": {
            "datasets": {"train": {"two_rooms": DATASET_CONFIG}},
            "collators": {"train": {"ac_video_jepa": {"collator_type": "ac_video_jepa"}}},
            "dataloader_configs": {"train": {"batch_size": 2}},
        }
    },
    "module": {
        "ac_video_jepa": {
            "module_type": "ac_video_jepa",
            "encoder": {
                "input_shape": [C, IMG, IMG],
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
                        "projector": {"enabled": False, "mlp_spec": None, "hidden_multiplier": 4},
                        "std_margin": 1.0,
                        "first_t_only": False,
                        "spatial_as_samples": False,
                    },
                    "covariance": {
                        "projector": {"enabled": False, "mlp_spec": None, "hidden_multiplier": 4},
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
                    "covariance": 1.0,
                },
            },
            "optimizer_configs": {"optimizer": {"optimizer_type": "adam", "lr": 1e-3}},
            "scheduler_configs": {},
        }
    },
    "trainer": {
        "max_steps": 1,
        "limit_val_batches": 0,
        "num_sanity_val_steps": 0,
        "accelerator": "cpu",
        "enable_checkpointing": False,
        "enable_progress_bar": False,
        "enable_model_summary": False,
    },
}


def test_ac_video_jepa_end_to_end_one_step():
    torch.manual_seed(0)
    objects = build_training_objects(RUN_CONFIG)
    trainer, module, datamodule = (
        objects["trainer"],
        objects["module"],
        objects["datamodule"],
    )
    trainer.fit(module, datamodule)
    assert trainer.state.finished
