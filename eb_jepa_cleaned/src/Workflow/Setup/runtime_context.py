"""
Runtime context construction.

`build_runtime_context` turns a resolved `setup` config (composed by Hydra) plus
the live environment into a plain `runtime_context` dict of runtime-determined
values: the resolved device, the applied reproducibility settings, and the
created run directories.

This is the boundary between static config and live runtime state (Decision 14):
Hydra composes the `setup` config; this builds the runtime_context from it. The
two are kept separate and passed to factories as distinct channels — they are
never merged into one config (logger handles, device, created paths, and probed
shapes are not serializable config).

Deferred (documented extension, not built here): credentials / wandb login
registration, and logger handles (loggers are built by AIML from config).
"""

from .device import resolve_device
from .paths import setup_paths
from .reproducibility import setup_reproducibility
from .user_credential import setup_user_credential
from .wandb_setup import setup_wandb

# Canonical shape of the `setup` config (authored as Hydra YAML at the
# experiment level). Hydra composes/merges defaults; the functions below also
# tolerate partial input so build_runtime_context is usable without Hydra.
DEFAULT_SETUP_CONFIG = {
    "device": "auto",
    "reproducibility": {"seed": 42, "deterministic": False, "cudnn_benchmark": False},
    "paths": {
        "project_root": None,
        "run_root": "runs",
        "experiment_name": "ac_video_jepa",
        "run_name": "default_run",
        "existing_run_dir_policy": "fail",
    },
    "user_credential": {
        "enabled": False,
        "path": "user_credential.yaml",
        "environment_variables": {"WANDB_API_KEY": ["wandb", "api_key"]},
    },
    "wandb": {
        "enabled": False,
        "mode": "online",
        "login": False,
        "api_key_env_var": "WANDB_API_KEY",
    },
}


def build_runtime_context(setup_config: dict | None = None) -> dict:
    """
    Build the runtime context from a (Hydra-resolved) setup config.

    Order matters: paths are resolved first (for the project root), then
    credentials are exported to the environment, then wandb (its login can read a
    key the credential step just exported).
    """
    setup_config = dict(setup_config or {})

    paths = setup_paths(setup_config.get("paths"))
    credentials = setup_user_credential(
        setup_config.get("user_credential"),
        project_root=paths["project_root"],
    )

    runtime_context = {
        "device": resolve_device(setup_config.get("device", "auto")),
        "reproducibility": setup_reproducibility(setup_config.get("reproducibility")),
        "paths": paths,
        "credentials": credentials,
    }
    if "wandb" in setup_config:
        runtime_context["wandb"] = setup_wandb(setup_config["wandb"])
    return runtime_context
