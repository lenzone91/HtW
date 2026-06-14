from copy import deepcopy

from ..Model.ac_video_jepa.configs import DEFAULT_AC_VIDEO_JEPA_BLOCKS_CONFIG
from ...Metrics.configs import DEFAULT_AC_VIDEO_JEPA_OBJECTIVE_CONFIG
from ...Rollouts.configs import DEFAULT_LATENT_ROLLOUT_CONFIG
from ...Training.Optimization.configs import (
    DEFAULT_ADAMW_CONFIG,
)
from ...Training.Schedulers.configs import DEFAULT_SCHEDULER_CONFIG


DEFAULT_AC_VIDEO_JEPA_MODULE_CONFIG = {
    "module_type": "ac_video_jepa",
    "blocks_config": deepcopy(DEFAULT_AC_VIDEO_JEPA_BLOCKS_CONFIG),
    "rollout_config": deepcopy(DEFAULT_LATENT_ROLLOUT_CONFIG),
    "objective_config": deepcopy(DEFAULT_AC_VIDEO_JEPA_OBJECTIVE_CONFIG),
    "optimizer_config": deepcopy(DEFAULT_ADAMW_CONFIG),
    "scheduler_config": deepcopy(DEFAULT_SCHEDULER_CONFIG),
}
