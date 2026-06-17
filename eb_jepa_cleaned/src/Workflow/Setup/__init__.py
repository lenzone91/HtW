"""
Workflow / Setup: builds the runtime context (device, reproducibility, run
directories) from a Hydra-composed `setup` config. Owns runtime-determined state,
kept separate from the static config (Decision 14/22).
"""

from .runtime_context import DEFAULT_SETUP_CONFIG, build_runtime_context

__all__ = ["build_runtime_context", "DEFAULT_SETUP_CONFIG"]
