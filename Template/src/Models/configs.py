#########################
# Default waveUnet config
#########################

DEFAULT_WAVEUNET_CONFIG = {

    # Small sanity-check model.
    # Faster inference is preferred over model capacity.
    "Fc": 4,
    "fd": 5,
    "fu": 5,
    "L": 2,
}


#########################
# Default models config
#########################

DEFAULT_MODELS_CONFIG = {
    "waveunet" : dict(DEFAULT_WAVEUNET_CONFIG)
}