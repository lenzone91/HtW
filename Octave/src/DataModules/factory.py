from ..Factory.base import BaseBuilder, BaseBuilderDispatcher
from ..Datasets.factory import build_datasets
from ..DataProcessing.factory import build_collator
from .base import BaseDataModule
from .datamodules import DefaultDataModule
from .configs import DEFAULT_DATAMODULE_CONFIG


#########################################################
# DataModuleBuilder
#########################################################


class DataModuleBuilder(BaseBuilder):
    """
    Build one LightningDataModule from a serializable config.

    The DataModule receives phase-indexed datasets, collators, and DataLoader
    configs.
    """

    def __init__(
        self,
        datamodule_class: type[BaseDataModule],
        default_config: dict = DEFAULT_DATAMODULE_CONFIG,
        strict: bool = True,
        check_default_keys: bool = True,
    ) -> None:
        super().__init__(
            default_config=default_config,
            strict=strict,
            check_default_keys=check_default_keys,
        )

        self.datamodule_class = datamodule_class

    def build_from_config(
        self,
        config: dict,
        runtime_context: dict | None = None,
    ) -> BaseDataModule:
        datasets = self.build_phase_datasets(
            phase_dataset_configs=config["datasets"],
            runtime_context=runtime_context,
        )

        collators = self.build_phase_collators(
            phase_collator_configs=config["collators"],
            runtime_context=runtime_context,
        )

        return self.datamodule_class(
            datasets=datasets,
            collators=collators,
            dataloader_configs=config["dataloader_configs"],
        )

    def build_phase_datasets(
        self,
        phase_dataset_configs: dict,
        runtime_context: dict | None = None,
    ) -> dict:
        datasets = {}

        for phase, dataset_config in phase_dataset_configs.items():
            if dataset_config is None:
                datasets[phase] = None
            else:
                datasets[phase] = self.build_single_phase_dataset(
                    dataset_config=dataset_config,
                    runtime_context=runtime_context,
                )

        return datasets

    def build_single_phase_dataset(
        self,
        dataset_config: dict,
        runtime_context: dict | None = None,
    ):
        built_datasets = build_datasets(
            dataset_configs=dataset_config,
            runtime_context=runtime_context,
        )

        if len(built_datasets) != 1:
            raise ValueError(
                "Each DataModule phase must define exactly one dataset config, "
                f"got {len(built_datasets)}."
            )

        return next(iter(built_datasets.values()))

    def build_phase_collators(
        self,
        phase_collator_configs: dict,
        runtime_context: dict | None = None,
    ) -> dict:
        collators = {}

        for phase, collator_config in phase_collator_configs.items():
            if collator_config is None:
                collators[phase] = None
            else:
                collators[phase] = build_collator(
                    collator_config=collator_config,
                    runtime_context=runtime_context,
                )

        return collators


#########################################################
# Registry
#########################################################


DATAMODULE_BUILDERS_REGISTRY = {
    "default": DataModuleBuilder(
        datamodule_class=DefaultDataModule,
        default_config=DEFAULT_DATAMODULE_CONFIG,
    ),
}


#########################################################
# Dispatcher
#########################################################


class DataModuleDispatcher(BaseBuilderDispatcher):
    """
    Build several DataModules from a named config dictionary.

    Expected config:
        datamodule_configs[datamodule_name] = datamodule_config
    """

    def __init__(
        self,
        builder_registry: dict = DATAMODULE_BUILDERS_REGISTRY,
        strict: bool = True,
    ) -> None:
        super().__init__(
            default_config={},
            builder_registry=builder_registry,
            strict=strict,
            check_default_keys=False,
        )


#########################################################
# Building wrappers
#########################################################


def build_datamodules(
    datamodule_configs: dict,
    runtime_context: dict | None = None,
) -> dict:
    """
    Build DataModules from a named DataModule config dictionary.
    """
    dispatcher = DataModuleDispatcher(
        strict=datamodule_configs.get("strict", True),
    )

    resolved_datamodule_configs = dict(datamodule_configs)
    resolved_datamodule_configs.pop("strict", None)

    return dispatcher(
        config=resolved_datamodule_configs,
        runtime_context=runtime_context,
    )


def build_datamodule(
    datamodule_configs: dict,
    runtime_context: dict | None = None,
) -> BaseDataModule:
    """
    Build exactly one DataModule from a named DataModule config dictionary.
    """
    datamodules = build_datamodules(
        datamodule_configs=datamodule_configs,
        runtime_context=runtime_context,
    )

    if len(datamodules) != 1:
        raise ValueError(
            "build_datamodule expects exactly one named DataModule config, "
            f"got {len(datamodules)}."
        )

    return next(iter(datamodules.values()))