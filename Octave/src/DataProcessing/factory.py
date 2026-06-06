from tqdm import tqdm

from ..Factory.base import BaseBuilder, BaseBuilderDispatcher
from .base import BaseBatchTransform, BaseCollator
from .transforms import AddNoiseAtSNR
from .collators import TSEWaveformCollator
from .configs import (
    DEFAULT_ADD_NOISE_AT_SNR_CONFIG,
    DEFAULT_TSE_WAVEFORM_COLLATOR_CONFIG,
)


##################################################
# TransformBuilder
##################################################


class TransformBuilder(BaseBuilder):
    """
    Build one batch transform instance from its keyword config.
    """

    def __init__(
        self,
        transform_class: type[BaseBatchTransform],
        default_config: dict,
        strict: bool = True,
        check_default_keys: bool = True,
    ) -> None:
        super().__init__(
            default_config=default_config,
            strict=strict,
            check_default_keys=check_default_keys,
        )

        self.transform_class = transform_class

    def build_from_config(
        self,
        config: dict,
        runtime_context: dict | None = None,
    ) -> BaseBatchTransform:
        return self.transform_class(**config)


##################################
# TransformBuilder registry
##################################


TRANSFORM_BUILDERS_REGISTRY = {
    "add_noise_at_snr": TransformBuilder(
        transform_class=AddNoiseAtSNR,
        default_config=DEFAULT_ADD_NOISE_AT_SNR_CONFIG,
    ),
}


##################################
# TransformDispatcher
##################################


class TransformDispatcher(BaseBuilderDispatcher):
    """
    Build several batch transforms from named transform configs.

    Expected format:
        transform_configs[transform_name] = transform_config

    The insertion order of the config dictionary defines the transform order.
    """

    def __init__(
        self,
        builder_registry: dict = TRANSFORM_BUILDERS_REGISTRY,
        strict: bool = True,
        show_progress: bool = False,
    ) -> None:
        super().__init__(
            default_config={},
            builder_registry=builder_registry,
            strict=strict,
            check_default_keys=False,
        )

        self.show_progress = show_progress

    def build_from_config(
        self,
        config: dict,
        runtime_context: dict | None = None,
    ) -> list[BaseBatchTransform]:
        transforms = []

        iterator = tqdm(
            config.items(),
            desc="Building batch transforms",
            disable=not self.show_progress,
        )

        for transform_name, transform_config in iterator:
            transform = self.build_one(
                object_name=transform_name,
                object_config=transform_config,
                runtime_context=runtime_context,
            )

            if transform is not None:
                transforms.append(transform)

        return transforms

    def handle_unknown_object(self, object_name: str) -> None:
        self.handle_error(
            f"Unknown batch transform '{object_name}'. "
            f"Available batch transforms are: {sorted(self.builder_registry.keys())}."
        )


##################################
# Transform building wrapper
##################################


def build_transforms(
    config: dict | None,
    runtime_context: dict | None = None,
    strict: bool = True,
    show_progress: bool = False,
) -> list[BaseBatchTransform]:
    """
    Build an ordered list of batch transforms.
    """
    if config is None:
        return []

    if len(config) == 0:
        return []

    dispatcher = TransformDispatcher(
        strict=strict,
        show_progress=show_progress,
    )

    return dispatcher(
        config=config,
        runtime_context=runtime_context,
    )


##################################
# CollatorBuilder
##################################


class CollatorBuilder(BaseBuilder):
    """
    Build one collator and equip it with its ordered batch transforms.
    """

    def __init__(
        self,
        collator_class: type[BaseCollator],
        default_config: dict,
        strict: bool = True,
        check_default_keys: bool = True,
        show_progress: bool = False,
    ) -> None:
        super().__init__(
            default_config=default_config,
            strict=strict,
            check_default_keys=check_default_keys,
        )

        self.collator_class = collator_class
        self.show_progress = show_progress

    def build_from_config(
        self,
        config: dict,
        runtime_context: dict | None = None,
    ) -> BaseCollator:
        transforms = build_transforms(
            config=config["transforms"],
            runtime_context=runtime_context,
            strict=config.get("strict", self.strict),
            show_progress=self.show_progress,
        )

        collator_config = config["collator_config"]

        return self.collator_class(
            transforms=transforms,
            **collator_config,
        )


##################################
# CollatorBuilder registry
##################################


COLLATOR_BUILDERS_REGISTRY = {
    "tse_waveform": CollatorBuilder(
        collator_class=TSEWaveformCollator,
        default_config=DEFAULT_TSE_WAVEFORM_COLLATOR_CONFIG,
    ),
}


##################################################
# CollatorDispatcher
##################################################


class CollatorDispatcher(BaseBuilderDispatcher):
    """
    Build collators from named collator configs.

    Expected format:
        collator_configs[collator_name] = collator_config
    """

    def __init__(
        self,
        builder_registry: dict = COLLATOR_BUILDERS_REGISTRY,
        strict: bool = True,
    ) -> None:
        super().__init__(
            default_config={},
            builder_registry=builder_registry,
            strict=strict,
            check_default_keys=False,
        )

    def handle_unknown_object(self, object_name: str) -> None:
        self.handle_error(
            f"Unknown collator '{object_name}'. "
            f"Available collators are: {sorted(self.builder_registry.keys())}."
        )


#############################################################################
# Collator building wrappers
#############################################################################


def build_collators(
    collator_configs: dict,
    runtime_context: dict | None = None,
    strict: bool = True,
) -> dict[str, BaseCollator]:
    """
    Build several collators from named collator configs.
    """
    dispatcher = CollatorDispatcher(strict=strict)

    return dispatcher(
        config=collator_configs,
        runtime_context=runtime_context,
    )


def build_collator(
    collator_config: dict,
    runtime_context: dict | None = None,
    strict: bool = True,
) -> BaseCollator:
    """
    Build one collator from a single-entry named collator config.
    """
    collators = build_collators(
        collator_configs=collator_config,
        runtime_context=runtime_context,
        strict=strict,
    )

    if len(collators) != 1:
        raise ValueError(
            "build_collator expects exactly one collator config, "
            f"got {len(collators)}."
        )

    return next(iter(collators.values()))