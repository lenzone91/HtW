from collections import namedtuple

import pytest
import torch

from Octave.src.Data.Datasets.two_rooms import (
    AC_VIDEO_JEPA_SAMPLE_KEYS,
    WallDatasetWrapper,
)


WallSample = namedtuple(
    "WallSample",
    [
        "states",
        "actions",
        "locations",
        "wall_x",
        "door_y",
    ],
)


def make_eb_sample() -> WallSample:
    return WallSample(
        states=torch.zeros(2, 17, 64, 64),
        actions=torch.zeros(2, 17),
        locations=torch.zeros(2, 17),
        wall_x=torch.tensor(32),
        door_y=torch.tensor(10),
    )


def test_wall_dataset_wrapper_adapts_eb_sample_to_semantic_dict() -> None:
    sample = WallDatasetWrapper.to_sample_dict(
        eb_sample=make_eb_sample(),
        index=3,
    )

    assert tuple(sample.keys()) == AC_VIDEO_JEPA_SAMPLE_KEYS
    assert sample["states"].shape == (2, 17, 64, 64)
    assert sample["actions"].shape == (2, 17)
    assert sample["locations"].shape == (2, 17)
    assert sample["wall_x"].shape == ()
    assert sample["door_y"].shape == ()
    assert sample["metadata"] == {
        "source": "eb_jepa.two_rooms.wall_dataset",
        "index": 3,
    }


def test_wall_dataset_wrapper_rejects_missing_required_sample_field() -> None:
    incomplete_sample = namedtuple(
        "IncompleteSample",
        [
            "states",
            "actions",
            "locations",
            "wall_x",
        ],
    )(
        states=torch.zeros(2, 17, 64, 64),
        actions=torch.zeros(2, 17),
        locations=torch.zeros(2, 17),
        wall_x=torch.tensor(32),
    )

    with pytest.raises(TypeError, match="missing required fields"):
        WallDatasetWrapper.to_sample_dict(eb_sample=incomplete_sample)
