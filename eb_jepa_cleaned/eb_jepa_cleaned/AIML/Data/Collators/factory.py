from .registry import COLLATOR_BUILDER


#############################################
# Collator building wrappers
#############################################

# Thin entrypoints over the shared collator builder. Concrete collators register
# themselves in their own object files. The current concrete collator is
# audio-specific (Phase 3), so none are imported here yet.


def build_collator(
    collator_config: dict,
    collator_name: str,
    runtime_context: dict | None = None,
):
    """
    Build one collator from one config and an explicit registered name.
    """
    return COLLATOR_BUILDER.build_one(
        config=collator_config,
        runtime_context=runtime_context,
        name=collator_name,
    )


def build_collators(
    collator_configs: dict,
    runtime_context: dict | None = None,
) -> dict:
    """
    Build collators from a named collator config dictionary.
    """
    return COLLATOR_BUILDER.build_named(
        configs=collator_configs,
        runtime_context=runtime_context,
    )
