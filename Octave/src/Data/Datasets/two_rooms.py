from typing import Any

from eb_jepa.datasets.two_rooms.wall_dataset import WallDataset, WallDatasetConfig


AC_VIDEO_JEPA_SAMPLE_KEYS = (
    "states",
    "actions",
    "locations",
    "wall_x",
    "door_y",
    "metadata",
)


class WallDatasetWrapper(WallDataset):
    """
    Octave adapter around EB-JEPA's Two Rooms WallDataset.

    EB-JEPA returns a WallSample named tuple. Octave datasets expose semantic
    dictionaries so downstream collators and modules can depend on explicit
    keys instead of tuple positions.
    """

    def __init__(self, config: WallDatasetConfig) -> None:
        super().__init__(config=config)

    def __getitem__(self, index: int) -> dict[str, Any]:
        eb_sample = super().__getitem__(index)
        return self.to_sample_dict(eb_sample=eb_sample, index=index)

    @staticmethod
    def to_sample_dict(eb_sample: Any, index: int | None = None) -> dict[str, Any]:
        missing_fields = [
            field_name
            for field_name in AC_VIDEO_JEPA_SAMPLE_KEYS
            if field_name != "metadata" and not hasattr(eb_sample, field_name)
        ]

        if missing_fields:
            raise TypeError(
                "EB-JEPA Two Rooms sample is missing required fields: "
                f"{missing_fields}."
            )

        metadata = {
            "source": "eb_jepa.two_rooms.wall_dataset",
        }

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
