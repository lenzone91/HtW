from copy import deepcopy

import pytest
import torch
from torch.utils.data import Dataset

from Octave.src.Data.Collators.ac_video_jepa_collator import AcVideoJepaCollator
from Octave.src.Data.DataModules.base import BaseDataModule
from Octave.src.Data.DataModules.configs import DEFAULT_AC_VIDEO_JEPA_DATAMODULE_CONFIG
from Octave.src.Data.DataModules import factory as datamodule_factory
from Octave.src.Data.DataModules.factory import build_ac_video_jepa_datamodule
from Octave.src.Data.DataModules.ac_video_jepa_datamodule import AcVideoJepaDataModule


class DummyDataset(Dataset):
    def __len__(self) -> int:
        return 2

    def __getitem__(self, index: int) -> dict:
        return {
            "states": torch.full((2, 17, 64, 64), float(index)),
            "actions": torch.full((2, 17), float(index)),
            "locations": torch.full((2, 17), float(index)),
            "wall_x": torch.tensor(index + 32),
            "door_y": torch.tensor(index + 10),
            "metadata": {"index": index},
        }


def make_base_datamodule() -> BaseDataModule:
    return BaseDataModule(
        datasets={
            "train": DummyDataset(),
            "val": None,
            "test": None,
        },
        collators={
            "train": AcVideoJepaCollator(),
            "val": None,
            "test": None,
        },
        dataloader_configs={
            "train": {"batch_size": 2, "shuffle": False},
            "val": None,
            "test": None,
        },
    )


def make_tiny_datamodule_config() -> dict:
    config = deepcopy(DEFAULT_AC_VIDEO_JEPA_DATAMODULE_CONFIG)
    config["datasets"]["train"] = {
        "dataset_type": "two_rooms",
        "device": "cpu",
        "size": 2,
        "batch_size": 1,
        "sample_length": 4,
        "n_steps": 8,
        "img_size": 32,
        "fix_wall_location": 16,
        "fix_door_location": 10,
    }
    config["dataloader_configs"]["train"] = {
        "batch_size": 2,
        "shuffle": False,
        "num_workers": 0,
        "drop_last": False,
    }
    return config


def test_base_datamodule_train_dataloader_returns_batch_dict() -> None:
    datamodule = make_base_datamodule()

    batch = next(iter(datamodule.train_dataloader()))

    assert batch["states"].shape == (2, 2, 17, 64, 64)
    assert batch["metadata"] == [{"index": 0}, {"index": 1}]


def test_base_datamodule_val_dataloader_returns_empty_list_when_disabled() -> None:
    datamodule = make_base_datamodule()

    assert datamodule.val_dataloader() == []


def test_base_datamodule_test_dataloader_returns_empty_list_when_disabled() -> None:
    datamodule = make_base_datamodule()

    assert datamodule.test_dataloader() == []


def test_base_datamodule_rejects_missing_collator_for_existing_dataset() -> None:
    with pytest.raises(KeyError, match="Missing collator"):
        BaseDataModule(
            datasets={"train": DummyDataset()},
            collators={},
            dataloader_configs={"train": {"batch_size": 2}},
        )


def test_base_datamodule_rejects_missing_dataloader_config() -> None:
    with pytest.raises(KeyError, match="Missing DataLoader config"):
        BaseDataModule(
            datasets={"train": DummyDataset()},
            collators={"train": AcVideoJepaCollator()},
            dataloader_configs={},
        )


def test_base_datamodule_rejects_persistent_workers_without_workers() -> None:
    with pytest.raises(ValueError, match="persistent_workers=True requires"):
        BaseDataModule(
            datasets={"train": DummyDataset()},
            collators={"train": AcVideoJepaCollator()},
            dataloader_configs={
                "train": {
                    "batch_size": 2,
                    "num_workers": 0,
                    "persistent_workers": True,
                },
            },
        )


def test_build_ac_video_jepa_datamodule_builds_datamodule_from_plain_config() -> None:
    datamodule = build_ac_video_jepa_datamodule(
        config=make_tiny_datamodule_config(),
    )

    assert isinstance(datamodule, AcVideoJepaDataModule)
    assert datamodule.datasets["train"].config.device == "cpu"
    assert isinstance(datamodule.collators["train"], AcVideoJepaCollator)


def test_build_ac_video_jepa_datamodule_train_dataloader_returns_ac_batch() -> None:
    datamodule = build_ac_video_jepa_datamodule(
        config=make_tiny_datamodule_config(),
    )

    batch = next(iter(datamodule.train_dataloader()))

    assert batch["states"].shape[0] == 2
    assert batch["actions"].shape[0] == 2
    assert batch["locations"].shape[0] == 2
    assert len(batch["metadata"]) == 2


def test_build_ac_video_jepa_datamodule_does_not_mutate_input_config() -> None:
    config = make_tiny_datamodule_config()
    original_config = deepcopy(config)

    build_ac_video_jepa_datamodule(config=config)

    assert config == original_config


def test_build_ac_video_jepa_datamodule_rejects_unknown_top_level_key() -> None:
    config = {
        **make_tiny_datamodule_config(),
        "unknown": {},
    }

    with pytest.raises(KeyError, match="Unknown DataModule config keys"):
        build_ac_video_jepa_datamodule(config=config)


def test_build_ac_video_jepa_datamodule_rejects_unknown_phase() -> None:
    config = make_tiny_datamodule_config()
    config["datasets"]["predict"] = None

    with pytest.raises(KeyError, match="Unknown phases"):
        build_ac_video_jepa_datamodule(config=config)


def test_build_ac_video_jepa_datamodule_rejects_persistent_workers_without_workers() -> None:
    config = make_tiny_datamodule_config()
    config["dataloader_configs"]["train"]["persistent_workers"] = True

    with pytest.raises(ValueError, match="persistent_workers=True requires"):
        build_ac_video_jepa_datamodule(config=config)


def test_build_ac_video_jepa_datamodule_rejects_cuda_dataset_when_unavailable(
    monkeypatch,
) -> None:
    monkeypatch.setattr(datamodule_factory.torch.cuda, "is_available", lambda: False)
    config = make_tiny_datamodule_config()
    config["datasets"]["train"]["device"] = "cuda"
    config["dataloader_configs"]["train"]["num_workers"] = 0

    with pytest.raises(RuntimeError, match="torch.cuda.is_available\\(\\) is False"):
        build_ac_video_jepa_datamodule(config=config)


def test_build_ac_video_jepa_datamodule_rejects_cuda_dataset_with_workers(
    monkeypatch,
) -> None:
    monkeypatch.setattr(datamodule_factory.torch.cuda, "is_available", lambda: True)
    config = make_tiny_datamodule_config()
    config["datasets"]["train"]["device"] = "cuda"
    config["dataloader_configs"]["train"]["num_workers"] = 1

    with pytest.raises(ValueError, match="CUDA dataset sampling with num_workers > 0"):
        build_ac_video_jepa_datamodule(config=config)
