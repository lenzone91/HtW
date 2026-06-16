from . import two_rooms  # noqa: F401
from .configs import DEFAULT_DATASETS_CONFIG, DEFAULT_TWO_ROOMS_DATASET_CONFIG
from .registry import DATASET_REGISTRY
from ...Workflow.Factory.builder import RegistryBuilder


def make_dataset_builder() -> RegistryBuilder:
    return RegistryBuilder(
        registry=DATASET_REGISTRY,
    )


DATASET_BUILDER = RegistryBuilder(
        registry=DATASET_REGISTRY,
    )


def build_dataset_from_config(
    dataset_config: dict,
    dataset_name: str,
    runtime_context: dict | None = None,
):
    builder = make_dataset_builder()

    return builder.build_one(
        config=dataset_config,
        runtime_context=runtime_context,
        name=dataset_name,
    )


def build_datasets(
    dataset_configs: dict | None = None,
    runtime_context: dict | None = None,
) -> dict:
    builder = make_dataset_builder()

    return builder.build_named(
        configs=dataset_configs or DEFAULT_DATASETS_CONFIG,
        runtime_context=runtime_context,
    )


def build_dataset(
    dataset_config: dict,
    runtime_context: dict | None = None,
):
    datasets = build_datasets(
        dataset_configs=dataset_config,
        runtime_context=runtime_context,
    )

    if datasets is None:
        return None

    if len(datasets) != 1:
        raise ValueError(
            "build_dataset expects exactly one dataset config, "
            f"got {len(datasets)}."
        )

    return next(iter(datasets.values()))


def build_two_rooms_dataset(
    config: dict | None = None,
    runtime_context: dict | None = None,
):
    return build_dataset_from_config(
        dataset_config=config or DEFAULT_TWO_ROOMS_DATASET_CONFIG,
        dataset_name="two_rooms",
        runtime_context=runtime_context,
    )
