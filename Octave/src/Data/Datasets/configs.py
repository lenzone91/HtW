from dataclasses import asdict
from copy import deepcopy

from eb_jepa.datasets.two_rooms.wall_dataset import WallDatasetConfig


DEFAULT_TWO_ROOMS_DATASET_CONFIG = asdict(WallDatasetConfig())


DEFAULT_DATASETS_CONFIG = {
    "two_rooms": deepcopy(DEFAULT_TWO_ROOMS_DATASET_CONFIG),
}
