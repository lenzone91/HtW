"""
Unit tests for AIML.Training.EarlyStoppings.factory.
"""

import pytest
from lightning.pytorch.callbacks import EarlyStopping

from eb_jepa_cleaned.AIML.Training.EarlyStoppings.factory import (
    build_early_stopping_callbacks,
)
from eb_jepa_cleaned.Workflow.Factory.errors import (
    RegistryError,
)


def test_build_early_stopping_callbacks():
    callbacks = build_early_stopping_callbacks(
        {
            "stagnation": {
                "early_stopping_type": "best_value_stagnation",
                "monitor": "val/loss",
            },
            "threshold": {
                "early_stopping_type": "threshold",
                "monitor": "val/loss",
            },
        }
    )

    assert len(callbacks) == 2
    assert all(isinstance(cb, EarlyStopping) for cb in callbacks)


def test_unknown_early_stopping_raises():
    with pytest.raises(RegistryError, match="Unknown early stopping"):
        build_early_stopping_callbacks({"x": {"early_stopping_type": "nope"}})
