from ....Workflow.Factory.builder import RegistryBuilder
from ....Workflow.Factory.registry import Registry, SubBuildDeclaration
from ..Collators.registry import COLLATOR_BUILDER
from ..Datasets.registry import DATASET_BUILDER


#############################################
# DataModule registry (anchor + cross-subsystem wiring)
#############################################

# This registry anchors the datamodule family and also declares the
# cross-subsystem sub-builds (datasets and collators are built per phase). It
# imports the dataset and collator builders, but never imports datamodule object
# files, so there is no decorator-registry cycle.


DATAMODULE_REGISTRY = Registry(object_family="datamodule")
DATAMODULE_BUILDER = RegistryBuilder(registry=DATAMODULE_REGISTRY)


#############################################
# Sub-builds
#############################################

# One named dataset and one named collator are built per phase (train/val/test).
# A phase whose config is None yields a None object for that phase.

DATAMODULE_SUB_BUILDS = (
    SubBuildDeclaration(
        source_key="datasets",
        target_key="datasets",
        builder=DATASET_BUILDER,
        build_method="phase_single_named",
    ),
    SubBuildDeclaration(
        source_key="collators",
        target_key="collators",
        builder=COLLATOR_BUILDER,
        build_method="phase_single_named",
    ),
)
