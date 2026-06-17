"""
Vendored `two_rooms` dataset and environment.

These modules (`wall_dataset`, `dot_dataset`, `env`, `utils`, `normalizer`) are
copied **verbatim** from EB-JEPA's `eb_jepa/datasets/two_rooms`, with the only
change being the internal imports rewritten to relative form. They are vendored
(not re-implemented) so this project stays standalone (no `eb_jepa` dependency,
Decision 30). Treat them as third-party code: prefer wrapping over editing.

Runtime deps (the `acvideo` extra): numpy, torch, scipy, gymnasium, pyyaml.

The Octave-style dataset adapter that exposes these as framework samples lives in
`AcVideoJEPA/Data/Datasets`, not here.
"""

from .wall_dataset import WallDataset, WallDatasetConfig

__all__ = ["WallDataset", "WallDatasetConfig"]
