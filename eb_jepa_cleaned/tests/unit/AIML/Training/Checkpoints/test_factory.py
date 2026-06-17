"""
Unit tests for AIML.Training.Checkpoints.factory.
"""

import pytest
from lightning.pytorch.callbacks import ModelCheckpoint

from eb_jepa_cleaned.AIML.Training.Checkpoints.factory import (
    build_checkpoint_callbacks,
)
from eb_jepa_cleaned.Workflow.Factory.errors import (
    RegistryError,
)


def test_build_checkpoint_callbacks():
    callbacks = build_checkpoint_callbacks(
        {
            "last": {"checkpoint_type": "last"},
            "best": {"checkpoint_type": "best_value", "monitor": "val/loss"},
        }
    )

    assert len(callbacks) == 2
    assert all(isinstance(cb, ModelCheckpoint) for cb in callbacks)


def test_unknown_checkpoint_raises():
    with pytest.raises(RegistryError, match="Unknown checkpoint"):
        build_checkpoint_callbacks({"x": {"checkpoint_type": "nope"}})
