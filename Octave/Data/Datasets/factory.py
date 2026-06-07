from eb_jepa.datasets.two_rooms.utils import update_config_from_yaml
from eb_jepa.datasets.two_rooms.wall_dataset import WallDatasetConfig
from Octave.Data.Datasets.two_rooms import WallDatasetWrapper

def build_two_rooms(env_name = "two_rooms",
                    config=None,
                    **kwargs):
    
    config = update_config_from_yaml(WallDatasetConfig, config)

    return WallDatasetWrapper(config, **kwargs)




