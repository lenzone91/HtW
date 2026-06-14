"""Loss-free rollout behavior for Octave runtime objects."""

from .factory import build_latent_rollout
from .latent_rollout import LatentRollout, LatentRolloutOutput

__all__ = [
    "LatentRollout",
    "LatentRolloutOutput",
    "build_latent_rollout",
]
