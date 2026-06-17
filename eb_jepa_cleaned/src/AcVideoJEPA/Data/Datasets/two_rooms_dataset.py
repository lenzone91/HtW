"""
Two-rooms dataset adapter.

Wraps the vendored `WallDataset` so it exposes semantic sample dictionaries
(explicit keys) instead of EB-JEPA's `WallSample` named tuple, and registers it
onto the AIML dataset registry. The build config is flat (the `WallDatasetConfig`
fields); the wrapper assembles the `WallDatasetConfig` from it.

Sample schema (JEPA is self-supervised, so it is not the generic
`{input, target}` convention): `{states, actions, locations, wall_x, door_y,
metadata}`. The collator and module read `states` / `actions`.
"""

from dataclasses import MISSING, fields
from typing import Any

from ....AIML.Data.Datasets.registry import DATASET_REGISTRY
from ..two_rooms import WallDataset, WallDatasetConfig

# Semantic keys the wrapper exposes (everything but `metadata` comes from the
# vendored WallSample).
AC_VIDEO_JEPA_SAMPLE_KEYS = (
    "states",
    "actions",
    "locations",
    "wall_x",
    "door_y",
    "metadata",
)

# Allow-list + canonical defaults, derived from the WallDatasetConfig dataclass.
# The builder does not merge defaults; omitted keys fall back to the dataclass.
DEFAULT_TWO_ROOMS_DATASET_CONFIG = {
    field.name: (field.default if field.default is not MISSING else None)
    for field in fields(WallDatasetConfig)
}


@DATASET_REGISTRY.register_class(
    name="two_rooms",
    default_config=DEFAULT_TWO_ROOMS_DATASET_CONFIG,
)
class TwoRoomsDataset(WallDataset):
    """Semantic-dictionary adapter around the vendored two-rooms WallDataset."""

    def __init__(self, **config_kwargs: Any) -> None:
        super().__init__(config=WallDatasetConfig(**config_kwargs))

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.to_sample_dict(super().__getitem__(index), index=index)

    @staticmethod
    def to_sample_dict(eb_sample: Any, index: int | None = None) -> dict[str, Any]:
        missing = [
            key
            for key in AC_VIDEO_JEPA_SAMPLE_KEYS
            if key != "metadata" and not hasattr(eb_sample, key)
        ]
        if missing:
            raise TypeError(
                f"two-rooms sample is missing required fields: {missing}."
            )

        metadata = {"source": "vendored.two_rooms.wall_dataset"}
        if index is not None:
            metadata["index"] = index

        return {
            "states": eb_sample.states,
            "actions": eb_sample.actions,
            "locations": eb_sample.locations,
            "wall_x": eb_sample.wall_x,
            "door_y": eb_sample.door_y,
            "metadata": metadata,
        }
