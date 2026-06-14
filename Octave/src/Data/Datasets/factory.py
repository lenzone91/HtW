from . import two_rooms  # noqa: F401
from .configs import DEFAULT_DATASETS_CONFIG, DEFAULT_TWO_ROOMS_DATASET_CONFIG
from .registry import DATASET_BUILDER


def build_dataset_from_config(
    dataset_config: dict,
    dataset_name: str,
    runtime_context: dict | None = None,
    strict: bool = True,
):
    DATASET_BUILDER.strict = strict

    return DATASET_BUILDER.build_one(
        config=dataset_config,
        runtime_context=runtime_context,
        name=dataset_name,
    )


def build_datasets(
    dataset_configs: dict | None = None,
    runtime_context: dict | None = None,
    strict: bool = True,
) -> dict:
    DATASET_BUILDER.strict = strict

    return DATASET_BUILDER.build_named(
        configs=dataset_configs or DEFAULT_DATASETS_CONFIG,
        runtime_context=runtime_context,
    )


def build_dataset(
    dataset_config: dict,
    runtime_context: dict | None = None,
    strict: bool = True,
):
    datasets = build_datasets(
        dataset_configs=dataset_config,
        runtime_context=runtime_context,
        strict=strict,
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
    strict: bool = True,
):
    return build_dataset_from_config(
        dataset_config=config or DEFAULT_TWO_ROOMS_DATASET_CONFIG,
        dataset_name="two_rooms",
        runtime_context=runtime_context,
        strict=strict,
    )