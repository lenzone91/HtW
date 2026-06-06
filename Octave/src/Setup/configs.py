#############################################
# Hardware setup
#############################################

# Hardware config expresses what the user expects from the runtime.
# It does not configure the Lightning Trainer directly.
# It only controls setup-time validation and diagnostics.

DEFAULT_HARDWARE_CONFIG = {
    # If True, hardware checks raise errors.
    # If False, impossible or suspicious requests only emit warnings.
    "strict": True,

    # Expected accelerator.
    # Supported setup-check values: None, "auto", "cpu", "cuda".
    # Lightning Trainer may still receive its own accelerator config elsewhere.
    "accelerator": None,

    # Expected number of CUDA devices.
    # Integer values are checked against available CUDA devices.
    # Other values are left to Lightning semantics.
    "devices": None,

    # Strong requirement that CUDA must be available.
    # This is stricter than accelerator="auto".
    "require_cuda": False,
}


#############################################
# Reproducibility setup
#############################################

# cuDNN is NVIDIA's CUDA Deep Neural Network library.
# PyTorch uses it internally to accelerate some GPU operations,
# especially convolutions.
#
# torch.backends.cudnn.benchmark = True lets cuDNN test several algorithms
# and select a fast one for the current tensor shapes.
# This can improve speed, but may reduce reproducibility.
#
# torch.use_deterministic_algorithms(True) asks PyTorch to use deterministic
# implementations when available.
# This improves reproducibility, but can slow training and may raise errors
# if some operations have no deterministic implementation.


DEFAULT_REPRODUCIBILITY_CONFIG = {
    # Balanced default: fixed seed, but no strict deterministic constraint.
    "seed": 42,
    "deterministic": False,
    "cudnn_benchmark": False,
}


DEFAULT_REPRODUCIBILITY_CONFIG_REPRODUCIBLE = {
    # Reproducibility-oriented behavior.
    # Prefer stable outputs over speed.
    "seed": 42,
    "deterministic": True,
    "cudnn_benchmark": False,
}


DEFAULT_REPRODUCIBILITY_CONFIG_PERFORMANCE = {
    # Performance-oriented behavior.
    # Keeps the seed fixed, but allows faster backend choices.
    "seed": 42,
    "deterministic": False,
    "cudnn_benchmark": True,
}





#############################################
# Environment setup
#############################################

# Environment config lists import names that must be available at runtime.
# Requirement names and import names may differ, so we keep import names explicit.
#
# This checks availability only.
# It does not install packages.

DEFAULT_ENVIRONMENT_CONFIG = {
    "required_imports": [
        # Notebook / development
        "ipykernel",

        # Scientific base
        "numpy",
        "pandas",
        "matplotlib",
        "tqdm",

        # Time series distances
        "dtaidistance",

        # Metrics
        "torchmetrics",

        # Loggers
        "wandb",

        # PyTorch stack
        "torch",
        "torchvision",
        "torchaudio",

        # Lightning
        "lightning",
    ],
}



#############################################
# Path setup
#############################################

# Path config expresses where runtime outputs should be created.
# It does not load configs.
# It does not save models, metrics, or logs.
# It only controls setup-time path preparation.

DEFAULT_PATHS_CONFIG = {
    # Root of the current idea / experiment workspace.
    # Should usually be injected as an absolute path by the config-generation script.
    # Root of the current runnable project.
    # If None, it is inferred from the launched config path.
    "project_root": None,

    # Root directory for all runs.
    # If None, paths.py defaults to idea_root / "results".
    "run_root": None,

    # High-level experiment name.
    "experiment_name": "default_experiment",

    # Name of the current run.
    # For WandB sweeps, this should usually be replaced by the WandB run id/name.
    "run_name": "default_run",

    # If None, the run is stored under single_runs/.
    # If provided, the run is stored under sweeps/sweep_name/.
    "sweep_name": None,

    # If True, an existing run directory is deleted before being recreated.
    "overwrite": False,
}


#############################################
# Logger registration setup
#############################################

# Logger registration config expresses which external logger backends
# should be prepared at setup time.
#
# It does not build Lightning loggers.
# It does not configure Trainer logging.
# It never stores API key values.

DEFAULT_LOGGER_REGISTRATION_CONFIG = {
    "wandb": {
        # WandB is the default supported online logger.
        "enabled": True,

        # If True, setup calls wandb.login().
        "login": True,

        # WandB runtime mode: "online", "offline", or "disabled".
        "mode": "online",

        # Name of the environment variable containing the WandB API key.
        "api_key_env_var": "WANDB_API_KEY",

        # If True, WandB registration failures raise errors.
        # If False, they are handled as warnings by handle_error.
        "strict": True,
    },
}


#############################################
# Data setup
#############################################

DEFAULT_DATA_CONFIG = {
    # If True, missing or invalid dataset roots raise errors.
    # If False, they only emit warnings through handle_error.
    "strict": True,

    # Named dataset roots resolved and validated at setup time.
    # Example:
    # "dataset_roots": {
    #     "librimix": "/path/to/LibriMix",
    # }
    "dataset_roots": {},
}


############################################
# User credential setup
############################################

# This is a dummy template that should be replaced by the actual user's info. See maintenance.md for more.

DEFAULT_USER_CREDENTIAL_CONFIG = {
    "enabled": False,
    "strict": True,
    "path": None,
    "environment_variables": {
        # Example:
        # "WANDB_API_KEY": ["wandb", "api_key"],
    },
}



########################################
# Default setup config
########################################

DEFAULT_SETUP_CONFIG = {
    "environment": DEFAULT_ENVIRONMENT_CONFIG,
    "hardware": DEFAULT_HARDWARE_CONFIG,
    "reproducibility": DEFAULT_REPRODUCIBILITY_CONFIG,
    "paths": DEFAULT_PATHS_CONFIG,
    "data": DEFAULT_DATA_CONFIG,
    "user_credential": DEFAULT_USER_CREDENTIAL_CONFIG,
    "logger_registration": DEFAULT_LOGGER_REGISTRATION_CONFIG,
}