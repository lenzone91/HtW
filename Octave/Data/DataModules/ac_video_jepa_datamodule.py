from .base import BaseDataModule
from ..Collators.base import BaseCollator
from ..Datasets.two_rooms import WallDatasetWrapper


class AcVideoJepaDataModule(BaseDataModule, ):
    def __init__(self, 
                 datasets: dict[str, WallDatasetWrapper | None], 
                 collators: dict[str, BaseCollator | None], 
                 dataloader_configs: dict[str, dict]) -> None:
        
        
        super().__init__(datasets, collators, dataloader_configs)


    