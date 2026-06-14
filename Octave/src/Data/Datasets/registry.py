from dataclasses import fields

from eb_jepa.datasets.two_rooms.wall_dataset import WallDatasetConfig

from .configs import DEFAULT_TWO_ROOMS_DATASET_CONFIG
from ...Workflow.Factory.builder import RegistryBuilder
from ...Workflow.Factory.registry import FieldResolution, Registry


WALL_DATASET_CONFIG_FIELDS = tuple(
    field.name
    for field in fields(WallDatasetConfig)
)


def make_two_rooms_default_config() -> dict:
    default_config = dict(DEFAULT_TWO_ROOMS_DATASET_CONFIG)
    default_config["dataset_type"] = "two_rooms"
    return default_config


def resolve_wall_dataset_config(
    config: dict,
    runtime_context: dict | None = None,
    strict: bool = True,
    **kwargs,
) -> WallDatasetConfig:
    config_kwargs = {
        key: value
        for key, value in config.items()
        if key in WALL_DATASET_CONFIG_FIELDS
    }

    return WallDatasetConfig(**config_kwargs)


DATASET_REGISTRY = Registry(object_family="dataset")

DATASET_BUILDER = RegistryBuilder(
    registry=DATASET_REGISTRY,
    type_field="dataset_type",
)