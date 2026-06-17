"""
AcVideoJEPA datasets: the two-rooms adapter over the vendored WallDataset.
Importing this subpackage registers it onto the AIML dataset registry.
"""

from . import factory  # noqa: F401  (registration side effect)
from .two_rooms_dataset import AC_VIDEO_JEPA_SAMPLE_KEYS, TwoRoomsDataset

__all__ = ["TwoRoomsDataset", "AC_VIDEO_JEPA_SAMPLE_KEYS"]
