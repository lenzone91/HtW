from . import optimizers  # noqa: F401
from .optimizers import ConfiguredOptimizerBuilder
from .registry import OPTIMIZER_REGISTRY
from ...Workflow.Factory.builder import RegistryBuilder


def build_optimizer(
    parameters,
    optimizer_config: dict,
    strict: bool = True,
):
    """
    Build one optimizer from parameters and a plain optimizer config.
    """
    optimizer_builder = build_optimizer_builder(
        optimizer_config=optimizer_config,
        strict=strict,
    )

    return optimizer_builder(parameters)


def build_optimizer_builder(
    optimizer_config: dict,
    strict: bool = True,
) -> ConfiguredOptimizerBuilder:
    """
    Build a delayed optimizer builder from a plain optimizer config.
    """
    builder = RegistryBuilder(
        registry=OPTIMIZER_REGISTRY,
        strict=strict,
        type_field="optimizer_type",
    )

    return builder.build_one(config=optimizer_config)
