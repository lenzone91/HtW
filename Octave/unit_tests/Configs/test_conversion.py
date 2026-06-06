"""
Pytest test module for Configs.conversion.

Pytest automatically discovers and runs functions whose names start with
``test_``. Therefore, no explicit test decorator is needed here.

The tests are intentionally written without fixtures or parametrization to keep
test setup explicit and easy to read.
"""

from pathlib import Path

import pytest

from Project_1_SepFormer_TSE_HF.idea_1_project_setup.src.Configs.conversion import load_config, save_config


#############################################
# Test data
#############################################

FIXTURES_DIR = Path(__file__).parent / "fixtures"


EXPECTED_CONVERSION_CONFIG = {
    "setup": {
        "seed": 123,
        "deterministic": True,
    },
    "trainer": {
        "max_epochs": 1,
        "accelerator": "cpu",
    },
}


#############################################
# Loading tests
#############################################

def test_load_yaml_config() -> None:
    """
    Test that a valid YAML config is loaded as the expected dictionary.
    """
    config = load_config(FIXTURES_DIR / "conversion_valid.yaml")

    assert isinstance(config, dict)
    assert config == EXPECTED_CONVERSION_CONFIG


def test_load_json_config() -> None:
    """
    Test that a valid JSON config is loaded as the expected dictionary.
    """
    config = load_config(FIXTURES_DIR / "conversion_valid.json")

    assert isinstance(config, dict)
    assert config == EXPECTED_CONVERSION_CONFIG


def test_load_toml_config() -> None:
    """
    Test that a valid TOML config is loaded as the expected dictionary.
    """
    config = load_config(FIXTURES_DIR / "conversion_valid.toml")

    assert isinstance(config, dict)
    assert config == EXPECTED_CONVERSION_CONFIG


def test_load_unsupported_extension_raises() -> None:
    """
    Test that loading an unsupported config extension fails explicitly.
    """
    with pytest.raises(ValueError):
        load_config(FIXTURES_DIR / "conversion_invalid.txt")


def test_load_non_dict_config_raises() -> None:
    """
    Test that loading a non-dictionary config fails explicitly.
    """
    with pytest.raises(TypeError):
        load_config(FIXTURES_DIR / "conversion_list.yaml")


#############################################
# Saving tests
#############################################

def test_save_then_load_yaml_config(tmp_path: Path) -> None:
    """
    Test that saving then loading a YAML config preserves the dictionary.
    """
    path = tmp_path / "saved_config.yaml"

    save_config(
        config=EXPECTED_CONVERSION_CONFIG,
        path=path,
    )

    loaded_config = load_config(path)

    assert loaded_config == EXPECTED_CONVERSION_CONFIG


def test_save_then_load_json_config(tmp_path: Path) -> None:
    """
    Test that saving then loading a JSON config preserves the dictionary.
    """
    path = tmp_path / "saved_config.json"

    save_config(
        config=EXPECTED_CONVERSION_CONFIG,
        path=path,
    )

    loaded_config = load_config(path)

    assert loaded_config == EXPECTED_CONVERSION_CONFIG


def test_save_unsupported_extension_raises(tmp_path: Path) -> None:
    """
    Test that saving to an unsupported config extension fails explicitly.
    """
    path = tmp_path / "saved_config.txt"

    with pytest.raises(ValueError):
        save_config(
            config=EXPECTED_CONVERSION_CONFIG,
            path=path,
        )


def test_save_non_dict_config_raises(tmp_path: Path) -> None:
    """
    Test that saving a non-dictionary config fails explicitly.
    """
    path = tmp_path / "invalid_config.yaml"

    with pytest.raises(TypeError):
        save_config(
            config=["setup", "trainer"],
            path=path,
        )