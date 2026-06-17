"""
AcVideoJEPA Lightning module: the registered `AcVideoJepaModule` (the JEPA step
over the latent rollout). Importing this subpackage registers it (and its build
dependencies) onto the AIML registries.
"""

from . import factory  # noqa: F401  (registration side effect)
from .ac_video_jepa_module import AcVideoJepaModule

__all__ = ["AcVideoJepaModule"]
