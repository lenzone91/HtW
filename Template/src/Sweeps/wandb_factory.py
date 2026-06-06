from copy import deepcopy


#############################################
# WandB sweep factory
#############################################

# This file contains config-building utilities for WandB sweeps.
#
# WandB chooses sampled hyperparameters.
# This file injects them into the project global config.
#
# It does not:
#   - create WandB sweeps;
#   - launch WandB agents;
#   - run training.


def build_wandb_trial_config(
    base_config: dict,
    sampled_parameters: dict,
    run_name: str | None = None,
) -> dict:
    """
    Build one project training config from WandB sampled parameters.

    WandB parameter names are expected to be dotted paths into base_config.

    Example:
        "trainer.max_epochs"
        "module.model_configs.model.hidden_channels"
    """
    trial_config = deepcopy(base_config)

    set_sampled_parameters(
        config=trial_config,
        sampled_parameters=sampled_parameters,
    )

    if run_name is not None:
        set_wandb_run_name(
            config=trial_config,
            run_name=run_name,
        )

    return trial_config


def set_sampled_parameters(
    config: dict,
    sampled_parameters: dict,
) -> None:
    """
    Inject sampled sweep parameters into a config.
    """
    for dotted_key, value in sampled_parameters.items():
        set_nested_config_value(
            config=config,
            dotted_key=dotted_key,
            value=value,
        )


def set_nested_config_value(
    config: dict,
    dotted_key: str,
    value,
) -> None:
    """
    Set a nested config value from a dotted key.
    """
    keys = dotted_key.split(".")
    current = config

    for key in keys[:-1]:
        current = get_next_config_level(
            current=current,
            dotted_key=dotted_key,
            key=key,
        )

    final_key = keys[-1]
    check_final_key_exists(
        current=current,
        dotted_key=dotted_key,
        final_key=final_key,
    )

    current[final_key] = value


def get_next_config_level(
    current: dict,
    dotted_key: str,
    key: str,
) -> dict:
    """
    Retrieve the next nested config level when setting a dotted key.
    """
    if key not in current:
        raise KeyError(
            f"Cannot set sweep parameter {dotted_key}: "
            f"missing intermediate key {key}."
        )

    next_level = current[key]

    if not isinstance(next_level, dict):
        raise TypeError(
            f"Cannot set sweep parameter {dotted_key}: "
            f"{key} does not point to a dictionary."
        )

    return next_level


def check_final_key_exists(
    current: dict,
    dotted_key: str,
    final_key: str,
) -> None:
    """
    Check that the final key already exists before overwriting it.
    """
    if final_key not in current:
        raise KeyError(
            f"Cannot set sweep parameter {dotted_key}: "
            f"missing final key {final_key}."
        )


def set_wandb_run_name(
    config: dict,
    run_name: str,
) -> None:
    """
    Store the WandB run name as the project run name.

    This assumes Setup/ uses config["setup"]["run_name"].
    """
    config["setup"]["run_name"] = run_name