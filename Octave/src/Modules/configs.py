
#################################################
# WaveUNet
#################################################

from ..Models.configs import DEFAULT_WAVEUNET_CONFIG
from ..Metrics.configs import DEFAULT_TSE_METRIC_SET_CONFIG, DEFAULT_TRAIN_TSE_METRIC_SET_CONFIG, DEFAULT_TSE_LOSS_CONFIG
from ..Optimization.configs import DEFAULT_ADAMW_CONFIG, DEFAULT_STEP_LR_CONFIG


DEFAULT_WAVEUNET_LIGHTNING_MODULE_CONFIG = {
    "module_type": "waveunet",

    "model_configs": {
        "waveunet": DEFAULT_WAVEUNET_CONFIG,
    },

    "train_metrics_config": DEFAULT_TRAIN_TSE_METRIC_SET_CONFIG,
    "val_metrics_config": DEFAULT_TSE_METRIC_SET_CONFIG,
    "test_metrics_config": DEFAULT_TSE_METRIC_SET_CONFIG,

    "loss_weights": DEFAULT_TSE_LOSS_CONFIG,

    "optimizer_configs": {
        "waveunet": {
            "optimizer_type": "adamw",
            **DEFAULT_ADAMW_CONFIG,
        },
    },

    "scheduler_configs": {
        "waveunet": {
            "scheduler_type": "step_lr",
            **DEFAULT_STEP_LR_CONFIG,
        },
    },

    "log_loss_ml_steps": ["train", "val"],
}


# Default lightning module config


DEFAULT_LIGHTNING_MODULE_CONFIG = {
    "waveunet": dict(DEFAULT_WAVEUNET_LIGHTNING_MODULE_CONFIG)
}