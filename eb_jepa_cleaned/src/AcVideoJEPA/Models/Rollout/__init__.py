"""
AcVideoJEPA latent rollout: roll the predictor forward in latent space
(autoregressive / parallel) and return a `LatentRolloutOutput`. Importing this
subpackage registers the rollout onto the local rollout registry.
"""

from . import factory  # noqa: F401  (registration side effect)
from .latent_rollout import LatentRollout
from .output import LatentRolloutOutput

__all__ = ["LatentRollout", "LatentRolloutOutput"]
