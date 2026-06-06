"""
Tests for data setup utilities.

This file validates dataset root resolution, dataset root existence checks,
and serializable data context creation.

Pytest automatically discovers and runs functions whose names start with
``test_``. Therefore, no explicit test decorator is needed here.
"""

from pathlib import Path

import pytest

from Project_1_SepFormer_TSE_HF.idea_1_project_setup.src.Setup.data import (
    check_dataset_root_exists,
    check_dataset_roots_exist,
    resolve_dataset_roots,
    setup_data,
)


#############################################
# Dataset root resolution
#############################################


def test_resolve_dataset_roots_resolves_all_paths(tmp_path: Path) -> None:
    dataset_root = tmp_path / "dataset"
    dataset_root.mkdir()

    resolved_roots = resolve_dataset_roots(
        dataset_roots={"toy_dataset": dataset_root},
    )

    assert resolved_roots == {
        "toy_dataset": dataset_root.resolve(),
    }


#############################################
# Dataset root checks
#############################################


def test_check_dataset_root_exists_accepts_existing_directory(tmp_path: Path) -> None:
    dataset_root = tmp_path / "dataset"
    dataset_root.mkdir()

    check_dataset_root_exists(
        dataset_name="toy_dataset",
        dataset_root=dataset_root,
    )


def test_check_dataset_root_exists_rejects_missing_directory(tmp_path: Path) -> None:
    dataset_root = tmp_path / "missing_dataset"

    with pytest.raises(RuntimeError):
        check_dataset_root_exists(
            dataset_name="toy_dataset",
            dataset_root=dataset_root,
            strict=True,
        )


def test_check_dataset_root_exists_rejects_non_directory(tmp_path: Path) -> None:
    dataset_root = tmp_path / "dataset.txt"
    dataset_root.write_text("not a directory", encoding="utf-8")

    with pytest.raises(RuntimeError):
        check_dataset_root_exists(
            dataset_name="toy_dataset",
            dataset_root=dataset_root,
            strict=True,
        )


def test_check_dataset_roots_exist_checks_all_roots(tmp_path: Path) -> None:
    dataset_root_0 = tmp_path / "dataset_0"
    dataset_root_1 = tmp_path / "dataset_1"
    dataset_root_0.mkdir()
    dataset_root_1.mkdir()

    check_dataset_roots_exist(
        dataset_roots={
            "dataset_0": dataset_root_0,
            "dataset_1": dataset_root_1,
        },
    )


#############################################
# Data setup
#############################################


def test_setup_data_returns_serializable_dataset_context(tmp_path: Path) -> None:
    dataset_root = tmp_path / "dataset"
    dataset_root.mkdir()

    context = setup_data(
        data_config={
            "strict": True,
            "dataset_roots": {
                "toy_dataset": dataset_root,
            },
        },
    )

    assert context == {
        "dataset_roots": {
            "toy_dataset": str(dataset_root.resolve()),
        },
    }