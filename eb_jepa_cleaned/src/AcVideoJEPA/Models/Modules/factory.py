"""
AcVideoJEPA module factory.

Importing this module registers `AcVideoJepaModule` onto the AIML
Lightning-module registry and ensures the objects its build resolvers need
(backbone models, rollout, objective metrics) are registered too. The module is
then built through the ordinary AIML `build_lightning_module`.
"""

from ..Backbones import factory as _backbones  # noqa: F401  (registers backbones)
from ..Rollout import factory as _rollout  # noqa: F401  (registers the rollout)
from ...Metrics import factory as _metrics  # noqa: F401  (registers objective metrics)
from . import ac_video_jepa_module  # noqa: F401  (registers AcVideoJepaModule)
from .ac_video_jepa_module import AcVideoJepaModule

__all__ = ["AcVideoJepaModule"]
