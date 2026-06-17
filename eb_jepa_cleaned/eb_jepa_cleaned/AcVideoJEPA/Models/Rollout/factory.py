"""
Rollout factory. Thin entrypoint over the local rollout builder; imports the
rollout object module for its registration side effect.
"""

from . import latent_rollout  # noqa: F401  (registers the latent rollout)
from .registry import ROLLOUT_BUILDER


def build_rollout(rollout_config: dict, runtime_context: dict | None = None):
    """Build one rollout, routed by the `rollout_type` field."""
    return ROLLOUT_BUILDER.build_one(
        config=rollout_config,
        runtime_context=runtime_context,
    )
