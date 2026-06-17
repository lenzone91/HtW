"""
Weights & Biases runtime registration.

Sets the `WANDB_MODE` environment variable (online / offline / disabled) and,
when requested, logs in with an API key read from the environment. This is the
runtime/credentials side of wandb; the actual `WandbLogger` is built by AIML from
the `loggers` config. `wandb` is imported lazily (only when login is needed), so
Workflow does not depend on it at import time.

Returns a serializable summary for the runtime context.
"""

import os

VALID_WANDB_MODES = {"online", "offline", "disabled"}


def setup_wandb(config: dict | None = None) -> dict:
    config = dict(config or {})
    enabled = config.get("enabled", False)
    if not enabled:
        return {"enabled": False}

    mode = config.get("mode", "online")
    if mode not in VALID_WANDB_MODES:
        raise ValueError(
            f"Invalid wandb mode '{mode}'. Expected one of {sorted(VALID_WANDB_MODES)}."
        )
    os.environ["WANDB_MODE"] = mode

    logged_in = False
    if config.get("login", False) and mode != "offline":
        api_key_env_var = config.get("api_key_env_var", "WANDB_API_KEY")
        api_key = os.environ.get(api_key_env_var)
        if not api_key:
            raise ValueError(
                f"wandb login requested but '{api_key_env_var}' is not set in the "
                "environment."
            )
        import wandb  # lazy: only needed to log in

        wandb.login(key=api_key)
        logged_in = True

    return {"enabled": True, "mode": mode, "logged_in": logged_in}
