DEFAULT_MODULE_LOADING_CONFIG = {
    "enabled": False,
    "type": "lightning_module",
    "checkpoint_path": None,
    "strict": True,
    "map_location": "cpu",
    "state_dict_key": "state_dict",
    "relative_to": "run_dir",
}


DEFAULT_LOADING_CONFIG = {
    "module": DEFAULT_MODULE_LOADING_CONFIG,
}
