from .registry import ADAPTATION_BUILDER


#############################################
# Data adaptation building wrappers
#############################################

# Thin entrypoints over the shared adaptation builder. Concrete adaptations
# register themselves in their own object files. The current ones are
# audio-specific (e.g. waveform -> spectrogram) and live in the Audio pillar
# (Phase 3), so none are imported here yet.


def build_adaptation(
    adaptation_config: dict,
    adaptation_name: str,
    runtime_context: dict | None = None,
):
    """
    Build one adaptation from one config and an explicit registered name.
    """
    return ADAPTATION_BUILDER.build_one(
        config=adaptation_config,
        runtime_context=runtime_context,
        name=adaptation_name,
    )


def build_adaptations(
    adaptation_configs: dict | None,
    runtime_context: dict | None = None,
) -> list:
    """
    Build an ordered list of adaptations from a named config dictionary.

    An empty or absent config means "no adaptation" and returns an empty list.
    Ordering follows the config insertion order.
    """
    if not adaptation_configs:
        return []

    adaptations = ADAPTATION_BUILDER.build_named(
        configs=adaptation_configs,
        runtime_context=runtime_context,
    )

    return list(adaptations.values())
