from .ac_video_jepa_datamodule import AcVideoJepaDataModule
from ..Datasets.factory import build_two_rooms

def build_two_rooms_datamodule(config: dict) -> AcVideoJepaDataModule:
    """
    Build a datamodule from a config dictionary.

    Expected config:
        {
            "datasets": dict,
            "collators": dict,
            "dataloader_configs": dict
        }
    """

    datasets_config = config["datasets"]
    train_set = build_two_rooms(config=datasets_config['train'])
    test_set = build_two_rooms(config=datasets_config['test'])
    datasets = {'train' : train_set,
                   'test' : test_set}

    collators_config = config["collators"]
    train_collator = collators_config["train"]
    test_collator = collators_config["test"]
    collators = {'train' : train_collator,
                 'test' : test_collator}
    
    

    return AcVideoJepaDataModule(
        datasets = datasets,
        collators = collators,
        dataloader_configs=config["dataloader_configs"],
    )