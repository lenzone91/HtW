"""
Pytest test module for project default configs.

This test dynamically scans src/**/configs.py files, imports them, collects
DEFAULT_*_CONFIG* dictionaries, and checks that they are YAML round-trip safe.
"""

from importlib import import_module
from pathlib import Path

from Project_1_SepFormer_TSE_HF.idea_1_project_setup.src.Configs.conversion import (
    load_config,
    save_config,
)


#############################################
# Project paths
#############################################

SRC_PACKAGE = "Project_1_SepFormer_TSE_HF.idea_1_project_setup.src"
SRC_DIR = Path(__file__).parents[2] / "src"


#############################################
# Discovery helpers
#############################################

def iter_config_modules() -> list[Path]:
    """
    Find every configs.py file under src/.
    """
    return sorted(SRC_DIR.rglob("configs.py"))


def config_path_to_module(config_path: Path) -> str:
    """
    Convert a src-relative configs.py path to its importable module path.
    """
    relative_path = config_path.relative_to(SRC_DIR).with_suffix("")
    module_suffix = ".".join(relative_path.parts)

    return f"{SRC_PACKAGE}.{module_suffix}"


def collect_default_configs() -> dict[str, dict]:
    """
    Collect every DEFAULT_*_CONFIG* dictionary from every configs.py module.
    """
    default_configs = {}

    for config_path in iter_config_modules():
        module_name = config_path_to_module(config_path)
        module = import_module(module_name)

        for object_name, object_value in vars(module).items():
            if not is_default_config_name(object_name):
                continue

            relative_config_path = config_path.relative_to(SRC_DIR)
            config_key = f"{relative_config_path}::{object_name}"

            default_configs[config_key] = object_value

    return default_configs


def is_default_config_name(name: str) -> bool:
    """
    Check whether a variable name follows the default config naming convention.
    """
    return name.startswith("DEFAULT_") and "_CONFIG" in name


def find_first_config_diff(
    expected: object,
    actual: object,
    path: str = "",
) -> str | None:
    """
    Return the first nested difference between two config-like objects.
    """
    if type(expected) is not type(actual):
        return (
            f"{path}: type mismatch "
            f"{type(expected).__name__} != {type(actual).__name__}"
        )

    if isinstance(expected, dict):
        expected_keys = set(expected.keys())
        actual_keys = set(actual.keys())

        if expected_keys != actual_keys:
            return (
                f"{path}: key mismatch "
                f"expected={expected_keys}, actual={actual_keys}"
            )

        for key in expected:
            key_path = f"{path}.{key}" if path else str(key)

            diff = find_first_config_diff(
                expected=expected[key],
                actual=actual[key],
                path=key_path,
            )

            if diff is not None:
                return diff

        return None

    if expected != actual:
        return (
            f"{path}: value mismatch "
            f"expected={expected!r}, actual={actual!r}"
        )

    return None


#############################################
# Project default config tests
#############################################



def test_project_default_configs_are_dicts() -> None:
    """
    Test that every discovered DEFAULT_*_CONFIG* object is a dictionary.
    """
    default_configs = collect_default_configs()

    assert len(default_configs) > 0

    for config_name, config in default_configs.items():
        assert isinstance(config, dict), config_name


def test_project_default_configs_are_yaml_round_trip_safe(tmp_path: Path) -> None:
    """
    Test that every discovered default config can be saved and reloaded.
    """
    default_configs = collect_default_configs()

    for config_name, config in default_configs.items():
        path = tmp_path / f"{config_name.replace('/', '_').replace(':', '_')}.yaml"

        save_config(
            config=config,
            path=path,
        )

        loaded_config = load_config(path)

        diff = find_first_config_diff(
            expected=config,
            actual=loaded_config,
        )

        assert diff is None, f"{config_name}: {diff}"