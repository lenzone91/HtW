from . import two_rooms as _two_rooms  # noqa: F401
from .configs import DEFAULT_DATASETS_CONFIG, DEFAULT_TWO_ROOMS_DATASET_CONFIG
from .registry import DATASET_REGISTRY
from .two_rooms import WallDatasetWrapper
from ...Workflow.Factory.builder import RegistryBuilder


def make_dataset_builder(strict: bool = True) -> RegistryBuilder:
    return RegistryBuilder(
        registry=DATASET_REGISTRY,
        strict=strict,
        type_field="dataset_type",
    )


def build_dataset(
    config: dict,
    runtime_context: dict | None = None,
    strict: bool = True,
) -> WallDatasetWrapper:
    builder = make_dataset_builder(strict=strict)
    return builder.build_one(
        config=config,
        runtime_context=runtime_context,
    )


def build_two_rooms_dataset(
    config: dict | None = None,
    runtime_context: dict | None = None,
    strict: bool = True,
) -> WallDatasetWrapper:
    builder = make_dataset_builder(strict=strict)
    return builder.build_one(
        config=config or DEFAULT_TWO_ROOMS_DATASET_CONFIG,
        runtime_context=runtime_context,
        name="two_rooms",
    )


def build_datasets(
    dataset_configs: dict | None = None,
    runtime_context: dict | None = None,
    strict: bool = True,
) -> dict[str, WallDatasetWrapper]:
    builder = make_dataset_builder(strict=strict)
    return builder.build_named(
        configs=dataset_configs or DEFAULT_DATASETS_CONFIG,
        runtime_context=runtime_context,
    )