"""
AcVideoJEPA data: the two-rooms dataset adapter, the collator, and the full
dataset -> collator -> DataLoader flow via the generic DefaultDataModule. Tiny
config keeps trajectory generation fast.
"""

import torch

# Register the dataset + collator onto the AIML data registries.
import src.AcVideoJEPA.Data.Collators  # noqa: F401
import src.AcVideoJEPA.Data.Datasets  # noqa: F401
from src.AcVideoJEPA.Data.Collators.ac_video_jepa_collator import (
    AcVideoJepaCollator,
)
from src.AcVideoJEPA.Data.Datasets.two_rooms_dataset import TwoRoomsDataset
from src.AIML.Data.Collators.factory import build_collator
from src.AIML.Data.DataModules.factory import build_datamodule
from src.AIML.Data.Datasets.factory import build_dataset

C, A, SAMPLE_LEN, HW = 2, 2, 3, 16
DATASET_CONFIG = {
    "size": 4,
    "val_size": 2,
    "n_steps": 6,
    "sample_length": SAMPLE_LEN,
    "img_size": HW,
    "batch_size": 2,
    "train": True,
    "device": "cpu",
}


def test_dataset_builds_and_yields_semantic_sample():
    dataset = build_dataset(DATASET_CONFIG, dataset_name="two_rooms")
    assert isinstance(dataset, TwoRoomsDataset)
    sample = dataset[0]
    assert set(sample) == {"states", "actions", "locations", "wall_x", "door_y", "metadata"}
    assert sample["states"].shape == (C, SAMPLE_LEN, HW, HW)
    assert sample["actions"].shape == (A, SAMPLE_LEN)
    assert sample["metadata"]["index"] == 0


def test_collator_stacks_into_semantic_batch():
    collator = build_collator({"collator_type": "ac_video_jepa"}, collator_name="ac_video_jepa")
    assert isinstance(collator, AcVideoJepaCollator)

    dataset = build_dataset(DATASET_CONFIG, dataset_name="two_rooms")
    batch = collator([dataset[0], dataset[1]])
    assert batch["states"].shape == (2, C, SAMPLE_LEN, HW, HW)
    assert batch["actions"].shape == (2, A, SAMPLE_LEN)
    assert isinstance(batch["metadata"], list) and len(batch["metadata"]) == 2


def test_collator_rejects_non_tensor_field():
    collator = AcVideoJepaCollator()
    bad = {
        "states": torch.zeros(C, SAMPLE_LEN, HW, HW),
        "actions": "not a tensor",
        "locations": torch.zeros(A, SAMPLE_LEN),
        "wall_x": torch.zeros(1),
        "door_y": torch.zeros(1),
        "metadata": {},
    }
    import pytest

    with pytest.raises(TypeError):
        collator([bad])


def test_full_data_flow_through_default_datamodule():
    datamodule = build_datamodule(
        {
            "default": {
                "datasets": {"train": {"two_rooms": DATASET_CONFIG}},
                "collators": {"train": {"ac_video_jepa": {"collator_type": "ac_video_jepa"}}},
                "dataloader_configs": {"train": {"batch_size": 2}},
            }
        }
    )
    batch = next(iter(datamodule.train_dataloader()))
    assert batch["states"].shape == (2, C, SAMPLE_LEN, HW, HW)
    assert batch["actions"].shape == (2, A, SAMPLE_LEN)
