from copy import deepcopy

import pytest

from eb_jepa.datasets.two_rooms.wall_dataset import WallDatasetConfig

from Octave.src.Data.Datasets.configs import DEFAULT_DATASETS_CONFIG
from Octave.src.Data.Datasets.factory import (
    build_datasets,
    build_two_rooms_dataset,
    make_dataset_builder,
)
from Octave.src.Data.Datasets.two_rooms import (
    WallDatasetWrapper,
    resolve_wall_dataset_config,
)


def make_tiny_two_rooms_config() -> dict:
    return {
        "device": "cpu",
        "size": 2,
        "val_size": 2,
        "batch_size": 1,
        "num_train_layouts": -1,
    }


def test_resolve_wall_dataset_config_builds_wall_dataset_config_from_plain_dict() -> None:
    wall_config = resolve_wall_dataset_config(config=make_tiny_two_rooms_config())

    assert isinstance(wall_config, WallDatasetConfig)
    assert wall_config.device == "cpu"
    assert wall_config.size == 2
    assert wall_config.batch_size == 1


def test_build_two_rooms_dataset_builds_wrapper_from_plain_dict() -> None:
    dataset = build_two_rooms_dataset(config=make_tiny_two_rooms_config())

    assert isinstance(dataset, WallDatasetWrapper)
    assert dataset.config.device == "cpu"
    assert len(dataset) == 2


def test_build_datasets_dispatches_two_rooms_dataset() -> None:
    dataset_configs = {
        "two_rooms": make_tiny_two_rooms_config(),
    }

    datasets = build_datasets(dataset_configs=dataset_configs)

    assert set(datasets) == {"two_rooms"}
    assert isinstance(datasets["two_rooms"], WallDatasetWrapper)


def test_build_datasets_does_not_mutate_input_config() -> None:
    dataset_configs = {
        "two_rooms": make_tiny_two_rooms_config(),
    }
    original_dataset_configs = deepcopy(dataset_configs)

    build_datasets(dataset_configs=dataset_configs)

    assert dataset_configs == original_dataset_configs


def test_dataset_builder_rejects_unknown_config_key() -> None:
    builder = make_dataset_builder()
    config = {
        **make_tiny_two_rooms_config(),
        "unknown_key": "bad",
    }

    with pytest.raises(RuntimeError, match="Invalid config keys"):
        builder.build_one(config=config, name="two_rooms")


def test_build_datasets_rejects_unknown_dataset_name() -> None:
    dataset_configs = {
        "unknown_dataset": make_tiny_two_rooms_config(),
    }

    with pytest.raises(RuntimeError, match="Unknown dataset"):
        build_datasets(dataset_configs=dataset_configs)


def test_default_datasets_config_is_plain_dictionary() -> None:
    assert isinstance(DEFAULT_DATASETS_CONFIG, dict)
    assert isinstance(DEFAULT_DATASETS_CONFIG["two_rooms"], dict)
