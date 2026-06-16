from copy import deepcopy

import pytest
import torch

from Octave.src.Data.Collators.ac_video_jepa_collator import AcVideoJepaCollator
from Octave.src.Data.Collators.base import BaseBatchTransform
from Octave.src.Data.Collators.configs import DEFAULT_AC_VIDEO_JEPA_COLLATOR_CONFIG
from Octave.src.Data.Collators.factory import build_ac_video_jepa_collator, build_collator


class AddFlagTransform(BaseBatchTransform):
    def transform(self, batch: dict) -> dict:
        batch["was_transformed"] = True
        return batch


def make_sample(index: int) -> dict:
    return {
        "states": torch.full((2, 17, 64, 64), float(index)),
        "actions": torch.full((2, 17), float(index + 1)),
        "locations": torch.full((2, 17), float(index + 2)),
        "wall_x": torch.tensor(index + 32),
        "door_y": torch.tensor(index + 10),
        "metadata": {
            "index": index,
        },
    }


def make_samples() -> list[dict]:
    return [
        make_sample(0),
        make_sample(1),
    ]


def test_ac_video_jepa_collator_stacks_tensor_fields() -> None:
    collator = AcVideoJepaCollator()

    batch = collator(make_samples())

    assert batch["states"].shape == (2, 2, 17, 64, 64)
    assert batch["actions"].shape == (2, 2, 17)
    assert batch["locations"].shape == (2, 2, 17)
    assert batch["wall_x"].shape == (2,)
    assert batch["door_y"].shape == (2,)


def test_ac_video_jepa_collator_preserves_metadata_as_list() -> None:
    collator = AcVideoJepaCollator()

    batch = collator(make_samples())

    assert batch["metadata"] == [
        {"index": 0},
        {"index": 1},
    ]


def test_ac_video_jepa_collator_applies_transforms_after_collation() -> None:
    collator = AcVideoJepaCollator(transforms=[AddFlagTransform()])

    batch = collator(make_samples())

    assert batch["was_transformed"] is True


def test_ac_video_jepa_collator_rejects_missing_required_key() -> None:
    samples = make_samples()
    samples[0].pop("states")

    collator = AcVideoJepaCollator()

    with pytest.raises(KeyError, match="missing AcVideoJepa keys"):
        collator(samples)


def test_ac_video_jepa_collator_rejects_non_tensor_value() -> None:
    samples = make_samples()
    samples[0]["states"] = "not a tensor"

    collator = AcVideoJepaCollator()

    with pytest.raises(TypeError, match="must be a torch.Tensor"):
        collator(samples)


def test_build_ac_video_jepa_collator_builds_default_collator() -> None:
    collator = build_ac_video_jepa_collator()

    assert isinstance(collator, AcVideoJepaCollator)


def test_build_collator_dispatches_ac_video_jepa_collator() -> None:
    config = deepcopy(DEFAULT_AC_VIDEO_JEPA_COLLATOR_CONFIG)

    collator = build_collator(config=config)

    assert isinstance(collator, AcVideoJepaCollator)


def test_build_collator_does_not_mutate_input_config() -> None:
    config = deepcopy(DEFAULT_AC_VIDEO_JEPA_COLLATOR_CONFIG)
    original_config = deepcopy(config)

    build_collator(config=config)

    assert config == original_config


def test_build_collators_rejects_unknown_collator_name() -> None:
    config = {
        "unknown": deepcopy(DEFAULT_AC_VIDEO_JEPA_COLLATOR_CONFIG),
    }

    with pytest.raises(RuntimeError, match="Unknown collator"):
        build_collator(config=config)


def test_build_collator_rejects_non_callable_transform() -> None:
    config = {
        **DEFAULT_AC_VIDEO_JEPA_COLLATOR_CONFIG,
        "transforms": ["not callable"],
    }

    with pytest.raises(TypeError):
        build_collator(config=config)
