###########################################
# Source separation dataset default config
###########################################

DEFAULT_SOURCE_SEPARATION_DATASET_CONFIG = {
    "path": None,
    "path_key": None,
    "canonical_dtype": "float32",
    "load_on_demand": False,
    "normalize_by_mixture_peak": True,
}


###########################################
# Default datasets config
###########################################

DEFAULT_DATASETS_CONFIG = {
    "source_separation": dict(DEFAULT_SOURCE_SEPARATION_DATASET_CONFIG),
}