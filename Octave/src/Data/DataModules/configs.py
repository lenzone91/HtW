from copy import deepcopy

from ..Collators.configs import DEFAULT_AC_VIDEO_JEPA_COLLATOR_CONFIG


DEFAULT_DATALOADER_CONFIG = {
    "batch_size": 2,
    "shuffle": False,
    "num_workers": 0,
    "drop_last": False,
}


DEFAULT_AC_VIDEO_JEPA_DATAMODULE_CONFIG = {
    "datasets": {
        "train": {
            "dataset_type": "two_rooms",
            "device": "cpu",
            "size": 2,
            "batch_size": 1,
        },
        "val": None,
        "test": None,
    },
    "collators": {
        "train": deepcopy(DEFAULT_AC_VIDEO_JEPA_COLLATOR_CONFIG),
        "val": None,
        "test": None,
    },
    "dataloader_configs": {
        "train": deepcopy(DEFAULT_DATALOADER_CONFIG),
        "val": None,
        "test": None,
    },
}
