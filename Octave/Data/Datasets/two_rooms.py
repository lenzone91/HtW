from eb_jepa.datasets.two_rooms.wall_dataset import WallDataset, WallDatasetConfig

class WallDatasetWrapper(WallDataset):
    def __init__(self, 
                 config = WallDatasetConfig,
                 **kwargs):
        super().__init__(config = config)