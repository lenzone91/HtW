"""
Tests for WandB sweep factory utilities.

This file validates dotted-path config injection and trial config creation.

Pytest automatically discovers and runs functions whose names start with
``test_``. Therefore, no explicit test decorator is needed here.
"""

from copy import deepcopy

import pytest

from Project_1_SepFormer_TSE_HF.idea_1_project_setup.src.Sweeps.wandb_factory import (
    build_wandb_trial_config,
    set_nested_config_value,
    set_sampled_parameters,
)


#############################################
# Helpers
#############################################


def make_base_config() -> dict:
    return {
        "setup": {
            "run_name": "base_run",
        },
        "trainer": {
            "max_epochs": 1,
            "log_every_n_steps": 10,
        },
        "module": {
            "model_configs": {
                "model": {
                    "hidden_channels": 16,
                },
            },
        },
    }


#############################################
# Nested config injection
#############################################


def test_set_nested_config_value_updates_existing_nested_key() -> None:
    config = make_base_config()

    set_nested_config_value(
        config=config,
        dotted_key="trainer.max_epochs",
        value=5,
    )

    assert config["trainer"]["max_epochs"] == 5


def test_set_nested_config_value_rejects_missing_intermediate_key() -> None:
    config = make_base_config()

    with pytest.raises(KeyError):
        set_nested_config_value(
            config=config,
            dotted_key="missing.max_epochs",
            value=5,
        )


def test_set_nested_config_value_rejects_non_dict_intermediate_key() -> None:
    config = make_base_config()

    with pytest.raises(TypeError):
        set_nested_config_value(
            config=config,
            dotted_key="trainer.max_epochs.value",
            value=5,
        )


def test_set_nested_config_value_rejects_missing_final_key() -> None:
    config = make_base_config()

    with pytest.raises(KeyError):
        set_nested_config_value(
            config=config,
            dotted_key="trainer.missing_key",
            value=5,
        )


#############################################
# Sampled parameter injection
#############################################


def test_set_sampled_parameters_updates_multiple_values() -> None:
    config = make_base_config()

    set_sampled_parameters(
        config=config,
        sampled_parameters={
            "trainer.max_epochs": 5,
            "module.model_configs.model.hidden_channels": 32,
        },
    )

    assert config["trainer"]["max_epochs"] == 5
    assert config["module"]["model_configs"]["model"]["hidden_channels"] == 32


#############################################
# Trial config construction
#############################################


def test_build_wandb_trial_config_does_not_mutate_base_config() -> None:
    base_config = make_base_config()
    original_base_config = deepcopy(base_config)

    trial_config = build_wandb_trial_config(
        base_config=base_config,
        sampled_parameters={
            "trainer.max_epochs": 5,
        },
    )

    assert base_config == original_base_config
    assert trial_config["trainer"]["max_epochs"] == 5


def test_build_wandb_trial_config_sets_run_name() -> None:
    trial_config = build_wandb_trial_config(
        base_config=make_base_config(),
        sampled_parameters={},
        run_name="wandb_trial_0",
    )

    assert trial_config["setup"]["run_name"] == "wandb_trial_0"