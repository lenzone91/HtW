from .registry import AUGMENTATION_BUILDER


#############################################
# Data augmentation building wrappers
#############################################

# Thin entrypoints over the shared augmentation builder. Concrete augmentations
# register themselves in their own object files. The current ones are
# audio-specific (e.g. add_noise_at_snr) and live in the Audio pillar (Phase 3),
# so none are imported here yet.


def build_augmentation(
    augmentation_config: dict,
    augmentation_name: str,
    runtime_context: dict | None = None,
):
    """
    Build one augmentation from one config and an explicit registered name.
    """
    return AUGMENTATION_BUILDER.build_one(
        config=augmentation_config,
        runtime_context=runtime_context,
        name=augmentation_name,
    )


def build_augmentations(
    augmentation_configs: dict | None,
    runtime_context: dict | None = None,
) -> list:
    """
    Build an ordered list of augmentations from a named config dictionary.

    An empty or absent config means "no augmentations" and returns an empty
    list. Ordering follows the config insertion order.
    """
    if not augmentation_configs:
        return []

    augmentations = AUGMENTATION_BUILDER.build_named(
        configs=augmentation_configs,
        runtime_context=runtime_context,
    )

    return list(augmentations.values())
