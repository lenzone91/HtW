"""
Tests for environment setup utilities.

This file validates import availability checks and environment context creation.
"""

import pytest

from Project_1_SepFormer_TSE_HF.idea_1_project_setup.src.Setup.environment import (
    check_required_import,
    check_required_imports,
    setup_environment,
)


#############################################
# Single import checks
#############################################


def test_check_required_import_accepts_existing_module() -> None:
    check_required_import("math")


def test_check_required_import_rejects_missing_module() -> None:
    with pytest.raises(ImportError):
        check_required_import("definitely_missing_module")


#############################################
# Multiple import checks
#############################################


def test_check_required_imports_accepts_existing_modules() -> None:
    check_required_imports(["math", "json"])


def test_check_required_imports_rejects_missing_module() -> None:
    with pytest.raises(ImportError):
        check_required_imports(["math", "definitely_missing_module"])


#############################################
# Environment setup
#############################################


def test_setup_environment_defaults_to_empty_required_imports() -> None:
    context = setup_environment()

    assert context == {"required_imports": ()}


def test_setup_environment_returns_required_imports_as_tuple() -> None:
    context = setup_environment(
        environment_config={"required_imports": ["math", "json"]},
    )

    assert context == {"required_imports": ("math", "json")}