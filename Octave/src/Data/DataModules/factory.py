import torch

from . import ac_video_jepa_datamodule  # noqa: F401
from .ac_video_jepa_datamodule import AcVideoJepaDataModule
from .configs import DEFAULT_AC_VIDEO_JEPA_DATAMODULE_CONFIG
from .registry import DATAMODULE_REGISTRY
from ...Workflow.Factory.builder import RegistryBuilder


VALID_PHASES = ("train", "val", "test")


def build_ac_video_jepa_datamodule(
    config: dict | None = None,
    runtime_context: dict | None = None,
) -> AcVideoJepaDataModule:
    return build_datamodule_from_config(
        datamodule_config=config or DEFAULT_AC_VIDEO_JEPA_DATAMODULE_CONFIG,
        datamodule_name="ac_video_jepa",
        runtime_context=runtime_context,
    )


def build_datamodule_from_config(
    datamodule_config: dict,
    datamodule_name: str,
    runtime_context: dict | None = None,
) -> AcVideoJepaDataModule:
    check_datamodule_runtime_semantics(datamodule_config)

    builder = RegistryBuilder(
        registry=DATAMODULE_REGISTRY,
    )

    return builder.build_one(
        config=datamodule_config,
        runtime_context=runtime_context,
        name=datamodule_name,
    )


def check_datamodule_runtime_semantics(config: dict) -> None:
    if not isinstance(config, dict):
        raise TypeError(
            "DataModule config must be a dictionary, "
            f"got {type(config).__name__}."
        )

    check_phase_sections(config)

    for phase in VALID_PHASES:
        dataset_config = get_single_phase_object_config(
            section=config["datasets"],
            phase=phase,
            section_name="datasets",
            optional=True,
        )

        dataloader_config = config["dataloader_configs"][phase]

        if dataset_config is None:
            continue

        check_phase_dataloader_config(
            phase=phase,
            dataloader_config=dataloader_config,
        )

        check_dataset_device_semantics(
            phase=phase,
            dataset_config=dataset_config,
            dataloader_config=dataloader_config,
        )


def check_phase_sections(config: dict) -> None:
    for section_name in ("datasets", "collators", "dataloader_configs"):
        if section_name not in config:
            raise KeyError(f"Missing DataModule config section '{section_name}'.")

        section = config[section_name]

        if not isinstance(section, dict):
            raise TypeError(
                f"DataModule config section '{section_name}' must be a dictionary, "
                f"got {type(section).__name__}."
            )

        if set(section) != set(VALID_PHASES):
            raise KeyError(
                f"DataModule section '{section_name}' must define phases "
                f"{sorted(VALID_PHASES)}, got {sorted(section)}."
            )


def get_single_phase_object_config(
    section: dict,
    phase: str,
    section_name: str,
    optional: bool,
) -> dict | None:
    phase_config = section[phase]

    if phase_config is None:
        if optional:
            return None

        raise ValueError(
            f"DataModule section '{section_name}' cannot be None "
            f"for phase '{phase}'."
        )

    if not isinstance(phase_config, dict):
        raise TypeError(
            f"DataModule section '{section_name}' for phase '{phase}' "
            f"must be a dictionary, got {type(phase_config).__name__}."
        )

    if len(phase_config) != 1:
        raise ValueError(
            f"DataModule section '{section_name}' for phase '{phase}' "
            f"must define exactly one named object, got {len(phase_config)}."
        )

    return next(iter(phase_config.values()))


def check_phase_dataloader_config(
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
