"""
Tests for generic error-handling utilities.

This file validates strict and non-strict error behavior.

Pytest automatically discovers and runs functions whose names start with
``test_``. Therefore, no explicit test decorator is needed here.
"""

import pytest

from Project_1_SepFormer_TSE_HF.idea_1_project_setup.src.Utils.error import (
    handle_error,
)


#############################################
# Strict mode
#############################################


def test_handle_error_raises_runtime_error_in_strict_mode() -> None:
    with pytest.raises(RuntimeError, match="dummy error"):
        handle_error(
            msg="dummy error",
            strict=True,
        )


#############################################
# Non-strict mode
#############################################


def test_handle_error_warns_in_non_strict_mode() -> None:
    with pytest.warns(UserWarning, match="dummy warning"):
        handle_error(
            msg="dummy warning",
            strict=False,
        )