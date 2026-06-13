DEFAULT_SETUP_CONFIG = {
    "paths": {
        "project_root": None,
        "run_root": "Octave/runs",
        "experiment_name": "ac_video_jepa",
        "run_name": "default_run",
        "overwrite": False,
    },
    "reproducibility": {
        "seed": 42,
        "deterministic": False,
        "cudnn_benchmark": False,
    },
    "logger_registration": {
        "wandb": {
            "enabled": False,
            "login": False,
            "mode": "online",
            "api_key_env_var": "WANDB_API_KEY",
            "strict": True,
        },
    },
}
