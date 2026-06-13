"""Lightning modules for Octave."""

from .ac_video_jepa_module import AcVideoJepaModule
from .factory import build_ac_video_jepa_module

__all__ = [
    "AcVideoJepaModule",
    "build_ac_video_jepa_module",
]
