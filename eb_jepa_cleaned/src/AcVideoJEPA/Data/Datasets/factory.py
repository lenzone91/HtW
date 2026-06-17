"""
AcVideoJEPA datasets factory. Importing this registers the two-rooms dataset
onto the AIML dataset registry; it is then built through `build_dataset(s)`.
"""

from . import two_rooms_dataset  # noqa: F401  (registers `two_rooms`)

REGISTERED_DATASETS = ("two_rooms",)
