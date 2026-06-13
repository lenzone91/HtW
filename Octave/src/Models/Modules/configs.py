from copy import deepcopy

from ..Model.ac_video_jepa.configs import DEFAULT_AC_VIDEO_JEPA_MODEL_CONFIG
from ...Training.Optimization.configs import (
    DEFAULT_ADAMW_CONFIG,
)
from ...Training.Schedulers.configs import DEFAULT_SCHEDULER_CONFIG


DEFAULT_AC_VIDEO_JEPA_UNROLL_CONFIG = {
    "nsteps": 2,
    "unroll_mode": "autoregressive",
    "ctxt_window_time": 1,
    "return_all_steps": False,
}


DEFAULT_AC_VIDEO_JEPA_MODULE_CONFIG = {
    "module_type": "ac_video_jepa",
    "model_config": deepcopy(DEFAULT_AC_VIDEO_JEPA_MODEL_CONFIG),
    "unroll_config": deepcopy(DEFAULT_AC_VIDEO_JEPA_UNROLL_CONFIG),
    "optimizer_config": deepcopy(DEFAULT_ADAMW_CONFIG),
    "scheduler_config": deepcopy(DEFAULT_SCHEDULER_CONFIG),
}
