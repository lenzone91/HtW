"""
Tests for reproducibility setup utilities.

This file validates seed checks, backend flag checks, and reproducibility
context creation.

Pytest automatically discovers and runs functions whose names start with
``test_``. Therefore, no explicit test decorator is needed here.
"""

import pytest
import torch

from Project_1_SepFormer_TSE_HF.idea_1_project_setup.src.Setup.reproducibility import (
    check_seed,
    set_cudnn_benchmark,
    set_deterministic_mode,
    setup_reproducibility,
)


#############################################
# Seed checks
#############################################


def test_check_seed_accepts_non_negative_integer() -> None:
    check_seed(42)


def test_check_seed_rejects_non_integer_seed() -> None:
    with pytest.raises(TypeError):
        check_seed("42")


def test_check_seed_rejects_negative_seed() -> None:
    with pytest.raises(ValueError):
        check_seed(-1)


#############################################
# Backend flag checks
#############################################


def test_set_deterministic_mode_rejects_non_bool() -> None:
    with pytest.raises(TypeError):
        set_deterministic_mode("True")


def test_set_cudnn_benchmark_rejects_non_bool() -> None:
    with pytest.raises(TypeError):
        set_cudnn_benchmark("False")


#############################################
# Reproducibility setup
#############################################


def test_setup_reproducibility_returns_context() -> None:
    context = setup_reproducibility(
        reproducibility_config={
            "seed": 42,
            "deterministic": False,
            "cudnn_benchmark": False,
        },
    )

    assert context == {
        "seed": 42,
        "deterministic": False,
        "cudnn_benchmark": False,
    }


def test_setup_reproducibility_sets_cudnn_benchmark() -> None:
    setup_reproducibility(
        reproducibility_config={
            "seed": 42,
            "deterministic": False,
            "cudnn_benchmark": True,
        },
    )

    assert torch.backends.cudnn.benchmark is True

    set_cudnn_benchmark(False)