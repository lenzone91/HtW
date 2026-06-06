import os

import wandb

from ..Utils.error import handle_error


#############################################
# Logger registration setup
#############################################

# This file centralizes runtime logger registration.
#
# It does not build Lightning loggers.
# It does not configure Trainer logging.
# It only prepares requested external logger backends when needed.


def setup_logger_registration(
    logger_registration_config: dict | None = None,
) -> dict:
    """
    Register requested external logger backends and return registration context.
    """
    if logger_registration_config is None:
        logger_registration_config = {}

    logger_registration_context = {}

    for logger_name, logger_config in logger_registration_config.items():
        logger_registration_context[logger_name] = setup_one_logger_registration(
            logger_name=logger_name,
            logger_config=logger_config,
        )

    return logger_registration_context


def setup_one_logger_registration(
    logger_name: str,
    logger_config: dict,
) -> dict:
    """
    Register one external logger backend.
    """
    if logger_name not in LOGGER_REGISTRATION_DISPATCHER:
        raise ValueError(f"Unknown logger registration type: {logger_name}")

    setup_function = LOGGER_REGISTRATION_DISPATCHER[logger_name]
    return setup_function(logger_config)


#############################################
# WandB registration
#############################################


def setup_wandb_registration(wandb_config: dict | None = None) -> dict:
    """
    Optionally login to WandB and return WandB registration context.
    """
    if wandb_config is None:
        wandb_config = {}

    enabled = wandb_config.get("enabled", False)
    login = wandb_config.get("login", False)
    mode = wandb_config.get("mode", "online")
    strict = wandb_config.get("strict", True)
    api_key_env_var = wandb_config.get("api_key_env_var", "WANDB_API_KEY")

    if not enabled:
        return build_wandb_context(
            enabled=False,
            logged_in=None,
            mode=mode,
            api_key_env_var=api_key_env_var,
            api_key_found=None,
        )

    logged_in = None
    api_key_found = os.environ.get(api_key_env_var) is not None

    if login:
        logged_in = login_wandb(
            api_key_env_var=api_key_env_var,
            strict=strict,
        )

    return build_wandb_context(
        enabled=True,
        logged_in=logged_in,
        mode=mode,
        api_key_env_var=api_key_env_var,
        api_key_found=api_key_found,
    )


def login_wandb(
        api_key_env_var: str,
        strict: bool,
    ) -> bool:
        api_key = os.environ.get(api_key_env_var)

        if api_key is None:
            handle_error(
                msg=f"WandB API key environment variable is missing: {api_key_env_var}.",
                strict=strict,
            )
            return False

        try:
            return bool(wandb.login(key=api_key))

        except Exception as error:
            handle_error(
                msg=f"WandB login failed using environment variable: {api_key_env_var}.",
                strict=strict,
            )
            return False


def build_wandb_context(
    enabled: bool,
    logged_in: bool | None,
    mode: str,
    api_key_env_var: str,
    api_key_found: bool | None,
) -> dict:
    """
    Build WandB registration context.

    The API key value is never stored in the runtime context.
    """
    return {
        "enabled": enabled,
        "logged_in": logged_in,
        "mode": mode,
        "api_key_env_var": api_key_env_var,
        "api_key_found": api_key_found,
    }


#############################################
# Dispatcher
#############################################

LOGGER_REGISTRATION_DISPATCHER = {
    "wandb": setup_wandb_registration,
}