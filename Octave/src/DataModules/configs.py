from ..Datasets.configs import DEFAULT_DATASETS_CONFIG
from ..DataProcessing.configs import DEFAULT_TSE_WAVEFORM_COLLATOR_CONFIGS


################################
# Default DataLoader configs
################################

DEFAULT_TRAIN_DATALOADER_CONFIG = {
    "batch_size": 8,
    "shuffle": True,
    "num_workers": 0,
    "pin_memory": False,
    "drop_last": True,
}

DEFAULT_VAL_DATALOADER_CONFIG = {
    "batch_size": 8,
    "shuffle": False,
    "num_workers": 0,
    "pin_memory": False,
    "drop_last": False,
}

DEFAULT_TEST_DATALOADER_CONFIG = {
    "batch_size": 8,
    "shuffle": False,
    "num_workers": 0,
    "pin_memory": False,
    "drop_last": False,
}


################################
# Default phase dataset configs
################################

DEFAULT_DATAMODULE_DATASET_CONFIGS = {
    "train": dict(DEFAULT_DATASETS_CONFIG),
    "val": dict(DEFAULT_DATASETS_CONFIG),
    "test": dict(DEFAULT_DATASETS_CONFIG),
}


################################
# Default phase collator configs
################################

DEFAULT_DATAMODULE_COLLATOR_CONFIGS = {
    "train": dict(DEFAULT_TSE_WAVEFORM_COLLATOR_CONFIGS),
    "val": dict(DEFAULT_TSE_WAVEFORM_COLLATOR_CONFIGS),
    "test": dict(DEFAULT_TSE_WAVEFORM_COLLATOR_CONFIGS),
}


################################
# Default DataModule config
################################

DEFAULT_DATAMODULE_CONFIG = {
    "strict": True,

    "datasets": dict(DEFAULT_DATAMODULE_DATASET_CONFIGS),

    "collators": dict(DEFAULT_DATAMODULE_COLLATOR_CONFIGS),

    "dataloader_configs": {
        "train": dict(DEFAULT_TRAIN_DATALOADER_CONFIG),
        "val": dict(DEFAULT_VAL_DATALOADER_CONFIG),
        "test": dict(DEFAULT_TEST_DATALOADER_CONFIG),
    },
}


######################################
# Wrapper for the dispatcher framework
######################################

DEFAULT_DATAMODULE_CONFIGS = {
    "default": dict(DEFAULT_DATAMODULE_CONFIG),
}