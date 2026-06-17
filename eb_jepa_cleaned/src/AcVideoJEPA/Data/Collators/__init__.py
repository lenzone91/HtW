"""
AcVideoJEPA collators: batch the semantic two-rooms samples. Importing this
subpackage registers the collator onto the AIML collator registry.
"""

from . import factory  # noqa: F401  (registration side effect)
from .ac_video_jepa_collator import AcVideoJepaCollator

__all__ = ["AcVideoJepaCollator"]
