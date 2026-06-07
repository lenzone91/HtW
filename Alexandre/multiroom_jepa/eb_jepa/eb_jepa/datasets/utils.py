from pathlib import Path

import torch
import yaml

from eb_jepa.datasets.two_rooms.utils import update_config_from_yaml
from eb_jepa.datasets.two_rooms.wall_dataset import WallDataset, WallDatasetConfig

DATASETS_DIR = Path(__file__).parent


def load_env_data_config(env_name: str, overrides: dict = None) -> dict:
    """Load base data config for an environment and apply overrides."""
    config_path = DATASETS_DIR / env_name / "data_config.yaml"
    with open(config_path) as f:
        base_config = yaml.safe_load(f)
    if overrides:
        base_config.update(overrides)
    return base_config


def init_data(env_name, cfg_data=None, **kwargs):
    """Initialize data loaders for the specified environment.

    Loads base config from eb_jepa/datasets/{env_name}/data_config.yaml
    and merges with any overrides from cfg_data.

    Args:
        env_name: Name of the environment ("two_rooms" or "multi_rooms").
        cfg_data: Configuration overrides for the dataset.

    Returns:
        Tuple of (train_loader, val_loader, config).
    """
    if env_name == "two_rooms":
        merged_cfg = load_env_data_config(env_name, cfg_data)
        config = update_config_from_yaml(WallDatasetConfig, merged_cfg)

        num_workers = merged_cfg.get("num_workers", 0)
        pin_mem = merged_cfg.get("pin_mem", False)
        persistent_workers = merged_cfg.get("persistent_workers", False) and num_workers > 0

        dset = WallDataset(config=config)
        loader = torch.utils.data.DataLoader(
            dset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_mem,
            drop_last=True,
            persistent_workers=persistent_workers,
        )

        val_dset = WallDataset(config=config)
        val_loader = torch.utils.data.DataLoader(
            val_dset,
            batch_size=4,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_mem,
            drop_last=True,
            persistent_workers=persistent_workers,
        )

        return loader, val_loader, config

    elif env_name == "multi_rooms":
        # Lazy import — eb_jepa stays importable without the environments package
        from environments.multiroom.config import MultiRoomConfig
        from environments.two_rooms.utils import update_config_from_yaml as _update_cfg
        from environments.multiroom.dataset import MultiRoomDataset

        base_cfg_path = Path(__file__).parent.parent.parent.parent / "environments" / "multiroom" / "data_config.yaml"
        import yaml

        with open(base_cfg_path) as f:
            base_cfg = yaml.safe_load(f)
        if cfg_data:
            base_cfg.update(cfg_data)

        config = _update_cfg(MultiRoomConfig, base_cfg)

        num_workers = base_cfg.get("num_workers", 0)
        pin_mem = base_cfg.get("pin_mem", False)
        persistent_workers = base_cfg.get("persistent_workers", False) and num_workers > 0

        dset = MultiRoomDataset(config=config)
        loader = torch.utils.data.DataLoader(
            dset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_mem,
            drop_last=True,
            persistent_workers=persistent_workers,
        )

        val_dset = MultiRoomDataset(config=config)
        val_loader = torch.utils.data.DataLoader(
            val_dset,
            batch_size=4,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_mem,
            drop_last=True,
            persistent_workers=persistent_workers,
        )

        return loader, val_loader, config

    else:
        raise ValueError(
            f"Unknown env: {env_name!r}. Supported: 'two_rooms', 'multi_rooms'."
        )
