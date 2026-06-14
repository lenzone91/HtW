from ...Workflow.Factory.builder import RegistryBuilder
from ...Workflow.Factory.registry import Registry, SubBuildDeclaration

from ..Collators import factory as collator_factory  # noqa: F401
from ..Collators.registry import COLLATOR_BUILDER
from ..Datasets import factory as dataset_factory  # noqa: F401
from ..Datasets.registry import DATASET_BUILDER


DATAMODULE_REGISTRY = Registry(object_family="datamodule")
DATAMODULE_BUILDER = RegistryBuilder(registry=DATAMODULE_REGISTRY)


DATASETS_SUB_BUILD = SubBuildDeclaration(
    source_key="datasets",
    target_key="datasets",
    builder=DATASET_BUILDER,
    build_method="phase_single_named",
)

COLLATORS_SUB_BUILD = SubBuildDeclaration(
    source_key="collators",
    target_key="collators",
    builder=COLLATOR_BUILDER,
    build_method="phase_single_named",
)