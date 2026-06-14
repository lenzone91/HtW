from copy import deepcopy

from ..Model.ac_video_jepa.configs import DEFAULT_AC_VIDEO_JEPA_COMPONENTS_CONFIG
from ...Metrics.Loss.configs import DEFAULT_WEIGHTED_METRIC_LOSS_CONFIG
from ...Metrics.MetricSets.configs import DEFAULT_AC_VIDEO_JEPA_METRIC_SET_CONFIG
from ...Rollouts.configs import DEFAULT_LATENT_ROLLOUT_CONFIG
from ...Training.Optimization.configs import DEFAULT_ADAMW_CONFIG
from ...Training.Schedulers.configs import DEFAULT_SCHEDULER_CONFIG


DEFAULT_AC_VIDEO_JEPA_MODULE_CONFIG = {
    "module_type": "ac_video_jepa",
    "components_config": deepcopy(DEFAULT_AC_VIDEO_JEPA_COMPONENTS_CONFIG),
    "rollout_config": deepcopy(DEFAULT_LATENT_ROLLOUT_CONFIG),
    "metric_set_config": deepcopy(DEFAULT_AC_VIDEO_JEPA_METRIC_SET_CONFIG),
    "loss_config": deepcopy(DEFAULT_WEIGHTED_METRIC_LOSS_CONFIG),
    "optimizer_config": deepcopy(DEFAULT_ADAMW_CONFIG),
    "scheduler_config": deepcopy(DEFAULT_SCHEDULER_CONFIG),
    "strict": True,
}


DEFAULT_AC_VIDEO_JEPA_MODULE_CONSTRUCTOR_CONFIG = {
    "module_type": "ac_video_jepa",
    "encoder": None,
    "action_encoder": None,
    "predictor": None,
    "encoder_shape": None,
    "rollout": None,
    "metric_set": None,
    "loss": None,
    "optimizer_builder": None,
    "scheduler_builder": None,
    "strict": True,
}
