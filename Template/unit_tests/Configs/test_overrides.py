"""
Pytest test module for Configs.overrides.

Pytest automatically discovers and runs functions whose names start with
``test_``. Therefore, no explicit test decorator is needed here.

The tests are intentionally written without fixtures or parametrization to keep
test setup explicit and easy to read.
"""

from pathlib import Path

import pytest

from Project_1_SepFormer_TSE_HF.idea_1_project_setup.src.Configs.conversion import load_config
from Project_1_SepFormer_TSE_HF.idea_1_project_setup.src.Configs.overrides import apply_overrides


#############################################
# Test data
#############################################

FIXTURES_DIR = Path(__file__).parent / "fixtures"


#############################################
# Override tests
#############################################

def test_apply_nested_overrides_dict() -> None:
    """
    Test that nested dictionary overrides are applied correctly.
    """
    base_config = load_config(FIXTURES_DIR / "overrides_base.yaml")
    overrides = load_config(FIXTURES_DIR / "overrides_dict.yaml")
    expected_config = load_config(FIXTURES_DIR / "overrides_expected.yaml")

    overridden_config = apply_overrides(
        config=base_config,
        overrides=overrides,
        strict=True,
    )

    assert overridden_config == expected_config


def test_apply_kwargs_top_level_override() -> None:
    """
    Test that keyword overrides replace top-level config entries.
    """
    base_config = load_config(FIXTURES_DIR / "overrides_base.yaml")

    overridden_config = apply_overrides(
        config=base_config,
        overrides=None,
        strict=True,
        trainer={
            "max_epochs": 5,
            "accelerator": "gpu",
        },
    )

    assert overridden_config["trainer"] == {
        "max_epochs": 5,
        "accelerator": "gpu",
    }


def test_apply_overrides_and_kwargs() -> None:
    """
    Test that dictionary overrides and keyword overrides are combined.
    """
    base_config = load_config(FIXTURES_DIR / "overrides_base.yaml")
    overrides = load_config(FIXTURES_DIR / "overrides_dict.yaml")

    overridden_config = apply_overrides(
        config=base_config,
        overrides=overrides,
        strict=True,
        trainer={
            "max_epochs": 5,
            "accelerator": "gpu",
        },
    )

    assert overridden_config["setup"]["seed"] == 456
    assert overridden_config["trainer"] == {
        "max_epochs": 5,
        "accelerator": "gpu",
    }


def test_apply_no_overrides_returns_independent_copy() -> None:
    """
    Test that missing overrides return an independent copy of the config.
    """
    base_config = load_config(FIXTURES_DIR / "overrides_base.yaml")

    overridden_config = apply_overrides(
        config=base_config,
        overrides=None,
        strict=True,
    )

    assert overridden_config == base_config
    assert overridden_config is not base_config


def test_apply_strict_unknown_key_raises() -> None:
    """
    Test that strict override application rejects unknown keys.
    """
    base_config = load_config(FIXTURES_DIR / "overrides_base.yaml")
    overrides = load_config(FIXTURES_DIR / "overrides_unknown_key.yaml")

    with pytest.raises(KeyError):
        apply_overrides(
            config=base_config,
            overrides=overrides,
            strict=True,
        )


def test_apply_non_strict_unknown_key_is_accepted() -> None:
    """
    Test that non-strict override application accepts unknown keys.
    """
    base_config = load_config(FIXTURES_DIR / "overrides_base.yaml")
    overrides = load_config(FIXTURES_DIR / "overrides_unknown_key.yaml")

    overridden_config = apply_overrides(
        config=base_config,
        overrides=overrides,
        strict=False,
    )

    assert overridden_config["unknown_section"] == {"value": 1}