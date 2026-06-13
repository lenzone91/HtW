from copy import deepcopy

import torch

from ..Collators.factory import build_collator
from ..Datasets.factory import build_datasets
from .ac_video_jepa_datamodule import AcVideoJepaDataModule
from .configs import DEFAULT_AC_VIDEO_JEPA_DATAMODULE_CONFIG


VALID_PHASES = ("train", "val", "test")


class AcVideoJepaDataModuleBuilder:
    """
    Build an AcVideoJepaDataModule from plain phase-keyed configs.
    """

    def __init__(
        self,
        default_config: dict | None = None,
        strict: bool = True,
    ) -> None:
        self.default_config = deepcopy(
            default_config
            if default_config is not None
            else DEFAULT_AC_VIDEO_JEPA_DATAMODULE_CONFIG
        )
        self.strict = strict

    def __call__(
        self,
        config: dict,
        runtime_context: dict | None = None,
    ) -> AcVideoJepaDataModule:
        prepared_config = self.prepare_config(config)

        datasets = self.build_phase_datasets(
            dataset_configs=prepared_config["datasets"],
            runtime_context=runtime_context,
        )
        collators = self.build_phase_collators(
            collator_configs=prepared_config["collators"],
        )

        return AcVideoJepaDataModule(
            datasets=datasets,
            collators=collators,
            dataloader_configs=prepared_config["dataloader_configs"],
        )

    def prepare_config(self, config: dict) -> dict:
        if not isinstance(config, dict):
            raise TypeError(
                "DataModule config must be a dictionary, "
                f"got {type(config).__name__}."
            )

        prepared_config = deepcopy(self.default_config)
        user_config = deepcopy(config)

        self.check_known_top_level_keys(user_config)

        for section_name, section_config in user_config.items():
            prepared_config[section_name] = self.merge_phase_section(
                default_section=prepared_config[section_name],
                user_section=section_config,
                section_name=section_name,
            )

        self.check_phase_sections(prepared_config)
        self.check_runtime_semantics(prepared_config)
        return prepared_config

    def merge_phase_section(
        self,
        default_section: dict,
        user_section: dict,
        section_name: str,
    ) -> dict:
        if not isinstance(user_section, dict):
            raise TypeError(
                f"DataModule config section '{section_name}' must be a dictionary, "
                f"got {type(user_section).__name__}."
            )

        unknown_phases = set(user_section) - set(VALID_PHASES)
        if unknown_phases:
            raise KeyError(
                f"Unknown phases in '{section_name}': {sorted(unknown_phases)}. "
                f"Allowed phases are: {sorted(VALID_PHASES)}."
            )

        merged_section = deepcopy(default_section)
        merged_section.update(deepcopy(user_section))
        return merged_section

    def build_phase_datasets(
        self,
        dataset_configs: dict,
        runtime_context: dict | None = None,
    ) -> dict:
        datasets = {}

        for phase, dataset_config in dataset_configs.items():
            if dataset_config is None:
                datasets[phase] = None
                continue

            dataset_config = deepcopy(dataset_config)
            dataset_type = dataset_config.pop("dataset_type", "two_rooms")
            datasets[phase] = build_datasets(
                dataset_configs={dataset_type: dataset_config},
                runtime_context=runtime_context,
                strict=self.strict,
            )[dataset_type]

        return datasets

    def build_phase_collators(self, collator_configs: dict) -> dict:
        collators = {}

        for phase, collator_config in collator_configs.items():
            if collator_config is None:
                collators[phase] = None
                continue

            collators[phase] = build_collator(config=collator_config)

        return collators

    def check_known_top_level_keys(self, config: dict) -> None:
        unknown_keys = set(config) - set(self.default_config)

        if not unknown_keys:
            return

        message = (
            "Unknown DataModule config keys: "
            f"{sorted(unknown_keys)}. "
            f"Allowed keys are: {sorted(self.default_config)}."
        )

        if self.strict:
            raise KeyError(message)

    def check_phase_sections(self, config: dict) -> None:
        for section_name in ("datasets", "collators", "dataloader_configs"):
            section = config[section_name]
            if set(section) != set(VALID_PHASES):
                raise KeyError(
                    f"DataModule section '{section_name}' must define phases "
                    f"{sorted(VALID_PHASES)}, got {sorted(section)}."
                )

    def check_runtime_semantics(self, config: dict) -> None:
        for phase in VALID_PHASES:
            dataset_config = config["datasets"][phase]
            dataloader_config = config["dataloader_configs"][phase]

            if dataset_config is None:
                continue

            self.check_phase_dataloader_config(
                phase=phase,
                dataloader_config=dataloader_config,
            )
            self.check_dataset_device_semantics(
                phase=phase,
                dataset_config=dataset_config,
                dataloader_config=dataloader_config,
            )

    def check_phase_dataloader_config(
        self,
        phase: str,
        dataloader_config: dict,
    ) -> None:
        if not isinstance(dataloader_config, dict):
            raise TypeError(
                f"DataLoader config for phase '{phase}' must be a dictionary, "
                f"got {type(dataloader_config).__name__}."
            )

        num_workers = dataloader_config.get("num_workers", 0)
        persistent_workers = dataloader_config.get("persistent_workers", False)

        if persistent_workers and num_workers <= 0:
            raise ValueError(
                "Invalid DataLoader config for phase "
                f"'{phase}': persistent_workers=True requires num_workers > 0."
            )

    def check_dataset_device_semantics(
        self,
        phase: str,
        dataset_config: dict,
        dataloader_config: dict,
    ) -> None:
        requested_device = str(dataset_config.get("device", "cpu"))
        num_workers = dataloader_config.get("num_workers", 0)

        if not requested_device.startswith("cuda"):
            return

        if not torch.cuda.is_available():
            raise RuntimeError(
                "Dataset config for phase "
                f"'{phase}' requested device='{requested_device}', but "
                "torch.cuda.is_available() is False."
            )

        if num_workers > 0:
            raise ValueError(
                "Invalid config for phase "
                f"'{phase}': CUDA dataset sampling with num_workers > 0 is unsafe. "
                "Use num_workers=0 or generate dataset samples on CPU."
            )


def build_ac_video_jepa_datamodule(
    config: dict | None = None,
    runtime_context: dict | None = None,
    strict: bool = True,
) -> AcVideoJepaDataModule:
    builder = AcVideoJepaDataModuleBuilder(strict=strict)
    return builder(
        config=config or DEFAULT_AC_VIDEO_JEPA_DATAMODULE_CONFIG,
        runtime_context=runtime_context,
    )
