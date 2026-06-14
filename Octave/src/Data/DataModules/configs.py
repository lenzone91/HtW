from copy import deepcopy

from ..Collators.configs import DEFAULT_AC_VIDEO_JEPA_COLLATOR_CONFIG
from ..Datasets.configs import DEFAULT_TWO_ROOMS_DATASET_CONFIG


DEFAULT_DATALOADER_CONFIG = {
    "batch_size": 2,
    "shuffle": False,
    "num_workers": 0,
    "drop_last": False,
}


DEFAULT_AC_VIDEO_JEPA_TRAIN_DATASET_CONFIG = deepcopy(
    DEFAULT_TWO_ROOMS_DATASET_CONFIG
)
DEFAULT_AC_VIDEO_JEPA_TRAIN_DATASET_CONFIG.update(
    {
        "device": "cpu",
        "size": 2,
        "batch_size": 1,
    }
)


DEFAULT_AC_VIDEO_JEPA_DATAMODULE_CONFIG = {
    "datasets": {
        "train": {
            "two_rooms": deepcopy(DEFAULT_AC_VIDEO_JEPA_TRAIN_DATASET_CONFIG),
        },
        "val": None,
        "test": None,
    },
    "collators": {
        "train": {
            "ac_video_jepa": deepcopy(DEFAULT_AC_VIDEO_JEPA_COLLATOR_CONFIG),
        },
        "val": None,
        "test": None,
    },
    "dataloader_configs": {
        "train": deepcopy(DEFAULT_DATALOADER_CONFIG),
        "val": None,
        "test": None,
    },
}
