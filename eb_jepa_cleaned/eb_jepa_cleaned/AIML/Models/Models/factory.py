from .registry import MODEL_BUILDER


#############################################
# Model building wrappers
#############################################

# Thin entrypoints over the shared model builder. Concrete models register
# themselves in their own object files. All current models (waveUnet, ...) are
# audio-specific (Phase 3), so none are imported here yet.


def build_model(
    model_config: dict,
    model_name: str,
    runtime_context: dict | None = None,
):
    """
    Build one model from one config and an explicit registered name.
    """
    return MODEL_BUILDER.build_one(
        config=model_config,
        runtime_context=runtime_context,
        name=model_name,
    )


def build_models(
    model_configs: dict,
    runtime_context: dict | None = None,
) -> dict:
    """
    Build several models from a named model config dictionary.
    """
    return MODEL_BUILDER.build_named(
        configs=model_configs,
        runtime_context=runtime_context,
    )
