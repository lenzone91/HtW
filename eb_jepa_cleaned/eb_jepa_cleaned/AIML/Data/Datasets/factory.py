from .registry import DATASET_BUILDER


#############################################
# Dataset building wrappers
#############################################

# Thin entrypoints over the shared dataset builder. Concrete datasets register
# themselves in their own object files; this module would import those modules
# for their registration side effect. There are no generic concrete datasets
# (all current datasets are audio-specific, Phase 3), so none are imported yet.


def build_dataset(
    dataset_config: dict,
    dataset_name: str,
    runtime_context: dict | None = None,
):
    """
    Build one dataset from one config and an explicit registered name.
    """
    return DATASET_BUILDER.build_one(
        config=dataset_config,
        runtime_context=runtime_context,
        name=dataset_name,
    )


def build_datasets(
    dataset_configs: dict,
    runtime_context: dict | None = None,
) -> dict:
    """
    Build datasets from a named dataset config dictionary.

    Outer keys are instance names (and registered names unless the entry routes
    by a type field).
    """
    return DATASET_BUILDER.build_named(
        configs=dataset_configs,
        runtime_context=runtime_context,
    )
