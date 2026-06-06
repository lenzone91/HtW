"""
Tests for early stopping callback factory utilities.

This file validates early stopping callback construction from named configs.

Pytest automatically discovers and runs functions whose names start with
``test_``. Therefore, no explicit test decorator is needed here.
"""

import pytest
from lightning.pytorch.callbacks import EarlyStopping

from Project_1_SepFormer_TSE_HF.idea_1_project_setup.src.EarlyStoppings.configs import (
    DEFAULT_EARLY_STOPPING_CONFIGS,
)
from Project_1_SepFormer_TSE_HF.idea_1_project_setup.src.EarlyStoppings.factory import (
    BestValueStagnationEarlyStoppingBuilder,
    DivergenceEarlyStoppingBuilder,
    FiniteValueEarlyStoppingBuilder,
    ThresholdEarlyStoppingBuilder,
    build_early_stopping_callbacks,
)


#############################################
# Builder tests
#############################################


def test_best_value_stagnation_builder_builds_early_stopping() -> None:
    callback = BestValueStagnationEarlyStoppingBuilder()(
        config={
            "monitor": "val/loss",
            "mode": "min",
            "patience": 10,
            "min_delta": 0.0,
        },
    )

    assert isinstance(callback, EarlyStopping)
    assert callback.monitor == "val/loss"
    assert callback.mode == "min"
    assert callback.patience == 10


def test_threshold_builder_builds_zero_patience_early_stopping() -> None:
    callback = ThresholdEarlyStoppingBuilder()(
        config={
            "monitor": "val/loss",
            "mode": "min",
            "stopping_threshold": 0.01,
        },
    )

    assert isinstance(callback, EarlyStopping)
    assert callback.monitor == "val/loss"
    assert callback.patience == 0


def test_divergence_builder_builds_zero_patience_early_stopping() -> None:
    callback = DivergenceEarlyStoppingBuilder()(
        config={
            "monitor": "val/loss",
            "mode": "min",
            "divergence_threshold": 1e3,
            "patience": 0,
        },
    )

    assert isinstance(callback, EarlyStopping)
    assert callback.monitor == "val/loss"
    assert callback.patience == 0


def test_finite_value_builder_builds_zero_patience_finite_check_early_stopping() -> None:
    callback = FiniteValueEarlyStoppingBuilder()(
        config={
            "monitor": "val/loss",
            "mode": "min",
        },
    )

    assert isinstance(callback, EarlyStopping)
    assert callback.monitor == "val/loss"
    assert callback.patience == 0
    assert callback.check_finite is True


#############################################
# Callback construction
#############################################


def test_build_early_stopping_callbacks_builds_default_callbacks() -> None:
    callbacks = build_early_stopping_callbacks(
        early_stopping_configs=DEFAULT_EARLY_STOPPING_CONFIGS,
    )

    assert len(callbacks) == 2
    assert all(isinstance(callback, EarlyStopping) for callback in callbacks)


def test_build_early_stopping_callbacks_rejects_unknown_early_stopping_type() -> None:
    with pytest.raises(RuntimeError):
        build_early_stopping_callbacks(
            early_stopping_configs={
                "unknown_callback": {
                    "early_stopping_type": "unknown_type",
                    "monitor": "val/loss",
                    "mode": "min",
                },
            },
        )


def test_build_early_stopping_callbacks_rejects_missing_early_stopping_type() -> None:
    with pytest.raises(RuntimeError):
        build_early_stopping_callbacks(
            early_stopping_configs={
                "missing_type_callback": {
                    "monitor": "val/loss",
                    "mode": "min",
                },
            },
        )