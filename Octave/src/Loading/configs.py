#############################################
# Loading default configs
#############################################

# Loading/ only restores weights into already-built objects.
#
# It does not:
#   - build models;
#   - build Lightning modules;
#   - resume Trainer state;
#   - restore optimizer or scheduler states.


DEFAULT_MODEL_LOADING_CONFIG = {
    "enabled": False,
    "type": "torch_model",
    "checkpoint_path": None,
    "strict": True,
    "map_location": "cpu",
    "state_dict_key": None,
    "relative_to": "idea_root"
}


DEFAULT_MODULE_LOADING_CONFIG = {
    "enabled": False,
    "type": "lightning_module",
    "checkpoint_path": None,
    "strict": True,
    "map_location": "cpu",
    "state_dict_key": "state_dict",
    "relative_to": "idea_root"
}


DEFAULT_LOADING_CONFIG = {
    "model": DEFAULT_MODEL_LOADING_CONFIG,
    "module": DEFAULT_MODULE_LOADING_CONFIG,
}