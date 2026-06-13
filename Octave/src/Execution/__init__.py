"""Execution entry points for Octave."""

from .train import run_training
from .validate import run_validation

__all__ = [
    "run_training",
    "run_validation",
]
