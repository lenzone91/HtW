"""
Pytest test module for Configs.merge.

Pytest automatically discovers and runs functions whose names start with
``test_``. Therefore, no explicit test decorator is needed here.

The tests are intentionally written without fixtures or parametrization to keep
test setup explicit and easy to read.
"""

from copy import deepcopy
from pathlib import Path

import pytest

from Project_1_SepFormer_TSE_HF.idea_1_project_setup.src.Configs.conversion import load_config
from Project_1_SepFormer_TSE_HF.idea_1_project_setup.src.Configs.merge import merge_configs


#############################################
# Test data
#############################################

FIXTURES_DIR = Path(__file__).parent / "fixtures"


#############################################
# Merge tests
#############################################

def test_merge_nested_override() -> None:
    """
    Test that nested override values replace matching nested base values.
    """
    base_config = load_config(FIXTURES_DIR / "merge_base.yaml")
    override_config = load_config(FIXTURES_DIR / "merge_override.yaml")
    expected_config = load_config(FIXTURES_DIR / "merge_expected.yaml")

    merged_config = merge_configs(
        base_config=base_config,
        override_config=override_config,
        strict=True,
    )

    assert merged_config == expected_config


def test_merge_does_not_mutate_base_config() -> None:
    """
    Test that merging does not modify the input base config.
    """
    base_config = load_config(FIXTURES_DIR / "merge_base.yaml")
    override_config = load_config(FIXTURES_DIR / "merge_override.yaml")
    initial_base_config = deepcopy(base_config)

    merge_configs(
        base_config=base_config,
        override_config=override_config,
        strict=True,
    )

    assert base_config == initial_base_config


def test_merge_none_override_returns_independent_copy() -> None:
    """
    Test that a missing override returns an independent copy of the base config.
    """
    base_config = load_config(FIXTURES_DIR / "merge_base.yaml")

    merged_config = merge_configs(
        base_config=base_config,
        override_config=None,
        strict=True,
    )

    assert merged_config == base_config
    assert merged_config is not base_config


def test_merge_strict_unknown_key_raises() -> None:
    """
    Test that strict merging rejects keys missing from the base config.
    """
    base_config = load_config(FIXTURES_DIR / "merge_base.yaml")
    override_config = load_config(FIXTURES_DIR / "merge_unknown_key.yaml")

    with pytest.raises(KeyError):
        merge_configs(
            base_config=base_config,
            override_config=override_config,
            strict=True,
        )


def test_merge_non_strict_unknown_key_is_accepted() -> None:
    """
    Test that non-strict merging accepts keys missing from the base config.

    This is useful for configs that intentionally forward extra keyword
    arguments to downstream objects.
    """
    base_config = load_config(FIXTURES_DIR / "merge_base.yaml")
    override_config = load_config(FIXTURES_DIR / "merge_unknown_key.yaml")

    merged_config = merge_configs(
        base_config=base_config,
        override_config=override_config,
        strict=False,
    )

    assert "unknown_section" in merged_config
    assert merged_config["unknown_section"] == {"value": 1}


#############################################
# Input validation tests
#############################################

def test_merge_non_dict_base_config_raises() -> None:
    """
    Test that base_config must be a dictionary.
    """
    with pytest.raises(TypeError):
        merge_configs(
            base_config=["setup", "trainer"],
            override_config={},
            strict=True,
        )


def test_merge_non_dict_override_config_raises() -> None:
    """
    Test that override_config must be a dictionary when provided.
    """
    base_config = load_config(FIXTURES_DIR / "merge_base.yaml")

    with pytest.raises(TypeError):
        merge_configs(
            base_config=base_config,
            override_config=["setup", "trainer"],
            strict=True,
        )