"""
Tests for the dataset factory.

This file validates dataset path resolution, dtype resolution, dispatcher behavior,
and config safety.
"""

from copy import deepcopy
from pathlib import Path

import pytest
import torch

from Project_1_SepFormer_TSE_HF.idea_1_project_setup.src.Datasets.factory import (
    DatasetBuilder,
    build_datasets,
)
from Project_1_SepFormer_TSE_HF.idea_1_project_setup.src.Datasets.configs import (
    DEFAULT_DATASETS_CONFIG,
    DEFAULT_SOURCE_SEPARATION_DATASET_CONFIG,
)
from Project_1_SepFormer_TSE_HF.idea_1_project_setup.src.Datasets.toy_dataset import (
    SourceSeparationDataset,
)


#############################################
# Helpers
#############################################


def create_empty_dataset_folder(tmp_path: Path) -> Path:
    dataset_folder = tmp_path / "toy_source_separation"
    dataset_folder.mkdir()

    return dataset_folder


def build_source_separation_builder() -> DatasetBuilder:
    return DatasetBuilder(
        dataset_class=SourceSeparationDataset,
        default_config=DEFAULT_SOURCE_SEPARATION_DATASET_CONFIG,
    )


#############################################
# Dtype resolution
#############################################


def test_dataset_builder_resolve_dtype_accepts_known_dtype() -> None:
    dtype = DatasetBuilder.resolve_dtype("float32")

    assert dtype == torch.float32


def test_dataset_builder_resolve_dtype_rejects_unknown_dtype() -> None:
    with pytest.raises(ValueError):
        DatasetBuilder.resolve_dtype("unknown_dtype")


#############################################
# Path resolution
#############################################


def test_dataset_builder_resolves_direct_path(tmp_path: Path) -> None:
    dataset_folder = create_empty_dataset_folder(tmp_path)
    builder = build_source_separation_builder()

    config = dict(DEFAULT_SOURCE_SEPARATION_DATASET_CONFIG)
    config["path"] = str(dataset_folder)

    resolved_path = builder.resolve_dataset_path(config=config)

    assert resolved_path == str(dataset_folder.resolve())


def test_dataset_builder_resolves_path_key_from_runtime_context(tmp_path: Path) -> None:
    dataset_folder = create_empty_dataset_folder(tmp_path)
    builder = build_source_separation_builder()

    config = dict(DEFAULT_SOURCE_SEPARATION_DATASET_CONFIG)
    config["path_key"] = "toy_dataset"

    runtime_context = {
        "data": {
            "dataset_roots": {
                "toy_dataset": str(dataset_folder),
            },
        },
    }

    resolved_path = builder.resolve_dataset_path(
        config=config,
        runtime_context=runtime_context,
    )

    assert resolved_path == str(dataset_folder.resolve())


def test_dataset_builder_rejects_path_and_path_key_together(tmp_path: Path) -> None:
    dataset_folder = create_empty_dataset_folder(tmp_path)
    builder = build_source_separation_builder()

    config = dict(DEFAULT_SOURCE_SEPARATION_DATASET_CONFIG)
    config["path"] = str(dataset_folder)
    config["path_key"] = "toy_dataset"

    with pytest.raises(RuntimeError):
        builder.resolve_dataset_path(config=config)


def test_dataset_builder_rejects_missing_path_information() -> None:
    builder = build_source_separation_builder()

    config = dict(DEFAULT_SOURCE_SEPARATION_DATASET_CONFIG)

    with pytest.raises(RuntimeError):
        builder.resolve_dataset_path(config=config)


#############################################
# build_datasets
#############################################


def test_build_datasets_builds_source_separation_dataset(tmp_path: Path) -> None:
    dataset_folder = create_empty_dataset_folder(tmp_path)

    dataset_configs = deepcopy(DEFAULT_DATASETS_CONFIG)
    dataset_configs["source_separation"]["path"] = str(dataset_folder)

    datasets = build_datasets(dataset_configs=dataset_configs)

    assert set(datasets.keys()) == {"source_separation"}
    assert isinstance(datasets["source_separation"], SourceSeparationDataset)


def test_build_datasets_rejects_unknown_dataset_name(tmp_path: Path) -> None:
    dataset_folder = create_empty_dataset_folder(tmp_path)

    dataset_configs = {
        "unknown_dataset": {
            **DEFAULT_SOURCE_SEPARATION_DATASET_CONFIG,
            "path": str(dataset_folder),
        },
    }

    with pytest.raises(RuntimeError):
        build_datasets(dataset_configs=dataset_configs)


def test_build_datasets_does_not_mutate_input_config(tmp_path: Path) -> None:
    dataset_folder = create_empty_dataset_folder(tmp_path)

    dataset_configs = deepcopy(DEFAULT_DATASETS_CONFIG)
    dataset_configs["source_separation"]["path"] = str(dataset_folder)
    original_dataset_configs = deepcopy(dataset_configs)

    build_datasets(dataset_configs=dataset_configs)

    assert dataset_configs == original_dataset_configs