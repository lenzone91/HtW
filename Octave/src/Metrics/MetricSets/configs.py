from copy import deepcopy

from ..Metrics.configs import DEFAULT_METRIC_CONFIGS


DEFAULT_AC_VIDEO_JEPA_METRIC_SET_CONFIG = {
    "set_type": "ac_video_jepa",
    "strict": True,
    "metric_to_input_names": None,
    "metrics": deepcopy(DEFAULT_METRIC_CONFIGS),
}