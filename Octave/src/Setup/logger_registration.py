import os


def setup_logger_registration(config: dict | None = None) -> dict:
    config = dict(config or {})
    wandb_config = dict(config.get("wandb", {}))

    return {
        "wandb": setup_wandb_registration(wandb_config),
    }


def setup_wandb_registration(config: dict) -> dict:
    enabled = config.get("enabled", False)
    mode = config.get("mode", "online")
    login = config.get("login", False)
    api_key_env_var = config.get("api_key_env_var", "WANDB_API_KEY")
    strict = config.get("strict", True)

    if not enabled:
        os.environ["WANDB_MODE"] = "disabled"
        return {
            "enabled": False,
            "mode": "disabled",
            "login_attempted": False,
        }

    if mode not in {"online", "offline", "disabled"}:
        raise ValueError(
            "wandb mode must be one of ['online', 'offline', 'disabled'], "
            f"got {mode!r}."
        )

    os.environ["WANDB_MODE"] = mode
    login_attempted = False

    if login and mode != "disabled":
        login_attempted = True
        api_key = os.environ.get(api_key_env_var)

        try:
            import wandb

            wandb.login(key=api_key)
        except Exception as error:
            if strict:
                raise RuntimeError("wandb login failed.") from error

    return {
        "enabled": True,
        "mode": mode,
        "login_attempted": login_attempted,
        "api_key_env_var": api_key_env_var,
    }
