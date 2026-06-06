"""
Tests for the DataModule factory.

This file validates named DataModule construction following the project factory
convention:
    registry[object_name] = object_builder
    config[object_name] = object_config
"""

from copy import deepcopy
from pathlib import Path

import pytest

from Project_1_SepFormer_TSE_HF.idea_1_project_setup.src.DataModules.factory import (
    DataModuleBuilder,
    build_datamodule,
    build_datamodules,
)
from Project_1_SepFormer_TSE_HF.idea_1_project_setup.src.DataModules.datamodules import (
    DefaultDataModule,
)
from Project_1_SepFormer_TSE_HF.idea_1_project_setup.src.DataModules.configs import (
    DEFAULT_DATAMODULE_CONFIG,
    DEFAULT_DATAMODULE_CONFIGS,
)
from Project_1_SepFormer_TSE_HF.idea_1_project_setup.src.DataProcessing.base import (
    BaseCollator,
)


#############################################
# Helpers
#############################################


def make_runtime_context(tmp_path: Path) -> dict:
    dataset_root = tmp_path / "toy_source_separation"
    dataset_root.mkdir()

    return {
        "data": {
            "dataset_roots": {
                "source_separation": str(dataset_root),
            },
        },
    }


def make_named_datamodule_config() -> dict:
    config = deepcopy(DEFAULT_DATAMODULE_CONFIGS)

    for phase in ["train", "val", "test"]:
        config["default"]["datasets"][phase]["source_separation"][
            "path_key"
        ] = "source_separation"

    return config


#############################################
# Dataset building
#############################################


def test_datamodule_builder_build_phase_datasets_builds_phase_datasets(
    tmp_path: Path,
) -> None:
    builder = DataModuleBuilder(
        datamodule_class=DefaultDataModule,
        default_config=DEFAULT_DATAMODULE_CONFIG,
    )
    config = make_named_datamodule_config()
    runtime_context = make_runtime_context(tmp_path)

    datasets = builder.build_phase_datasets(
        phase_dataset_configs=config["default"]["datasets"],
        runtime_context=runtime_context,
    )

    assert set(datasets.keys()) == {"train", "val", "test"}


def test_datamodule_builder_build_phase_datasets_preserves_none_dataset() -> None:
    builder = DataModuleBuilder(
        datamodule_class=DefaultDataModule,
        default_config=DEFAULT_DATAMODULE_CONFIG,
    )

    datasets = builder.build_phase_datasets(
        phase_dataset_configs={"train": None},
    )

    assert datasets == {"train": None}


#############################################
# Collator building
#############################################


def test_datamodule_builder_build_phase_collators_builds_phase_collators() -> None:
    builder = DataModuleBuilder(
        datamodule_class=DefaultDataModule,
        default_config=DEFAULT_DATAMODULE_CONFIG,
    )

    collators = builder.build_phase_collators(
        phase_collator_configs=DEFAULT_DATAMODULE_CONFIG["collators"],
    )

    assert set(collators.keys()) == {"train", "val", "test"}
    assert all(isinstance(collator, BaseCollator) for collator in collators.values())


def test_datamodule_builder_build_phase_collators_preserves_none_collator() -> None:
    builder = DataModuleBuilder(
        datamodule_class=DefaultDataModule,
        default_config=DEFAULT_DATAMODULE_CONFIG,
    )

    collators = builder.build_phase_collators(
        phase_collator_configs={"train": None},
    )

    assert collators == {"train": None}


#############################################
# Named DataModule construction
#############################################


def test_build_datamodules_builds_named_default_datamodule(tmp_path: Path) -> None:
    config = make_named_datamodule_config()
    runtime_context = make_runtime_context(tmp_path)

    datamodules = build_datamodules(
        datamodule_configs=config,
        runtime_context=runtime_context,
    )

    assert set(datamodules.keys()) == {"default"}
    assert isinstance(datamodules["default"], DefaultDataModule)


def test_build_datamodule_unwraps_single_named_datamodule(tmp_path: Path) -> None:
    config = make_named_datamodule_config()
    runtime_context = make_runtime_context(tmp_path)

    datamodule = build_datamodule(
        datamodule_configs=config,
        runtime_context=runtime_context,
    )

    assert isinstance(datamodule, DefaultDataModule)


def test_build_datamodule_rejects_multiple_named_datamodules(tmp_path: Path) -> None:
    config = make_named_datamodule_config()
    config["second_default"] = deepcopy(config["default"])
    runtime_context = make_runtime_context(tmp_path)

    with pytest.raises(RuntimeError):
        build_datamodule(
            datamodule_configs=config,
            runtime_context=runtime_context,
        )


def test_build_datamodules_rejects_unknown_datamodule_name(tmp_path: Path) -> None:
    config = make_named_datamodule_config()
    config["unknown"] = config.pop("default")
    runtime_context = make_runtime_context(tmp_path)

    with pytest.raises(RuntimeError):
        build_datamodules(
            datamodule_configs=config,
            runtime_context=runtime_context,
        )


def test_build_datamodules_forwards_runtime_context_to_dataset_factory(
    tmp_path: Path,
) -> None:
    config = make_named_datamodule_config()
    runtime_context = make_runtime_context(tmp_path)

    datamodule = build_datamodule(
        datamodule_configs=config,
        runtime_context=runtime_context,
    )

    train_dataset = datamodule.datasets["train"]

    assert train_dataset.path_to_folder == str(
        Path(runtime_context["data"]["dataset_roots"]["source_separation"]).resolve()
    )


def test_build_datamodules_does_not_mutate_input_config(tmp_path: Path) -> None:
    config = make_named_datamodule_config()
    original_config = deepcopy(config)
    runtime_context = make_runtime_context(tmp_path)

    build_datamodules(
        datamodule_configs=config,
        runtime_context=runtime_context,
    )

    assert config == original_config