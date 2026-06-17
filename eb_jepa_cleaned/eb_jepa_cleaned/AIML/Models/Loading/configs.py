#############################################
# Loading default configs
#############################################

# Reference templates documenting the expected loading_config shape. Loading
# only restores weights into already-built objects.
#
# The `relative_to` / runtime-context-rooted path resolution from Project_2 is
# deferred to the Setup migration (Decision 22), so it is not present here yet.


DEFAULT_MODEL_LOADING_CONFIG = {
    "enabled": False,
    "type": "torch_model",
    "checkpoint_path": None,
    "strict": True,
    "map_location": "cpu",
    "state_dict_key": None,
}


DEFAULT_MODULE_LOADING_CONFIG = {
    "enabled": False,
    "type": "lightning_module",
    "checkpoint_path": None,
    "strict": True,
    "map_location": "cpu",
    "state_dict_key": "state_dict",
}


DEFAULT_LOADING_CONFIG = {
    "model": dict(DEFAULT_MODEL_LOADING_CONFIG),
    "module": dict(DEFAULT_MODULE_LOADING_CONFIG),
}
