from copy import deepcopy
from dataclasses import fields

from eb_jepa.datasets.two_rooms.wall_dataset import WallDatasetConfig

from .configs import DEFAULT_DATASETS_CONFIG, DEFAULT_TWO_ROOMS_DATASET_CONFIG
from .two_rooms import WallDatasetWrapper


WALL_DATASET_CONFIG_FIELDS = {
    field.name
    for field in fields(WallDatasetConfig)
}


class TwoRoomsDatasetBuilder:
    """
    Build the AcVideoJepa Two Rooms dataset from a plain dictionary config.
    """

    def __init__(
        self,
        default_config: dict | None = None,
        strict: bool = True,
    ) -> None:
        self.default_config = deepcopy(
            default_config
            if default_config is not None
            else DEFAULT_TWO_ROOMS_DATASET_CONFIG
        )
        self.strict = strict

    def __call__(
        self,
        config: dict,
        runtime_context: dict | None = None,
    ) -> WallDatasetWrapper:
        prepared_config = self.prepare_config(config)
        wall_config = self.build_wall_dataset_config(prepared_config)
        return WallDatasetWrapper(config=wall_config)

    def prepare_config(self, config: dict) -> dict:
        if not isinstance(config, dict):
            raise TypeError(
                "Two Rooms dataset config must be a dictionary, "
                f"got {type(config).__name__}."
            )

        prepared_config = deepcopy(self.default_config)
        user_config = deepcopy(config)

        self.check_known_keys(user_config)
        prepared_config.update(user_config)

        return prepared_config

    def build_wall_dataset_config(self, config: dict) -> WallDatasetConfig:
        config_kwargs = {
            key: value
            for key, value in config.items()
            if key in WALL_DATASET_CONFIG_FIELDS
        }
        return WallDatasetConfig(**config_kwargs)

    def check_known_keys(self, config: dict) -> None:
        unknown_keys = set(config) - set(self.default_config)

        if not unknown_keys:
            return

        message = (
            "Unknown Two Rooms dataset config keys: "
            f"{sorted(unknown_keys)}. "
            f"Allowed keys are: {sorted(self.default_config)}."
        )

        if self.strict:
            raise KeyError(message)


DATASET_BUILDERS_REGISTRY = {
    "two_rooms": TwoRoomsDatasetBuilder,
}


def build_two_rooms_dataset(
    config: dict,
    runtime_context: dict | None = None,
    strict: bool = True,
) -> WallDatasetWrapper:
    builder = TwoRoomsDatasetBuilder(strict=strict)
    return builder(config=config, runtime_context=runtime_context)


def build_datasets(
    dataset_configs: dict,
    runtime_context: dict | None = None,
    strict: bool = True,
) -> dict[str, WallDatasetWrapper]:
    if not isinstance(dataset_configs, dict):
        raise TypeError(
            "Dataset configs must be a dictionary, "
            f"got {type(dataset_configs).__name__}."
        )

    resolved_configs = deepcopy(dataset_configs)
    datasets = {}

    for dataset_name, dataset_config in resolved_configs.items():
        if dataset_name not in DATASET_BUILDERS_REGISTRY:
            raise KeyError(
                f"Unknown dataset '{dataset_name}'. "
                f"Available datasets are: {sorted(DATASET_BUILDERS_REGISTRY)}."
            )

        builder_class = DATASET_BUILDERS_REGISTRY[dataset_name]
        builder = builder_class(strict=strict)
        datasets[dataset_name] = builder(
            config=dataset_config,
            runtime_context=runtime_context,
        )

    return datasets
