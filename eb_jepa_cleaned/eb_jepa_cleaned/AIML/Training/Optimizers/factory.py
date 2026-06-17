import torch

from . import optimizers  # noqa: F401  (registers SGD/Adam/AdamW)
from .registry import OPTIMIZER_BUILDER


#############################################
# Optimizer building wrappers
#############################################


def check_matching_parameter_groups(
    optimizer_configs: dict,
    parameter_groups: dict,
) -> None:
    """
    Check that every optimizer config has a matching parameter group.
    """
    if not isinstance(parameter_groups, dict):
        raise TypeError(
            f"parameter_groups must be a dictionary, got {type(parameter_groups)}."
        )

    missing_names = set(optimizer_configs.keys()) - set(parameter_groups.keys())

    if missing_names:
        raise KeyError(
            f"Missing parameter groups for {sorted(missing_names)}. "
            f"Available parameter groups are: {sorted(parameter_groups.keys())}."
        )


def build_optimizers(
    parameter_groups: dict,
    optimizer_configs: dict,
    runtime_context: dict | None = None,
) -> dict[str, torch.optim.Optimizer]:
    """
    Build one optimizer per named parameter group.
    """
    check_matching_parameter_groups(
        optimizer_configs=optimizer_configs,
        parameter_groups=parameter_groups,
    )

    return {
        optimizer_name: OPTIMIZER_BUILDER.build_one(
            config=optimizer_config,
            runtime_context=runtime_context,
            params=parameter_groups[optimizer_name],
        )
        for optimizer_name, optimizer_config in optimizer_configs.items()
    }


def build_optimizers_from_models(
    models: dict[str, torch.nn.Module],
    optimizer_configs: dict,
    runtime_context: dict | None = None,
) -> dict[str, torch.optim.Optimizer]:
    """
    Build optimizers directly from named models, using each model's parameters.
    """
    parameter_groups = {
        model_name: model.parameters() for model_name, model in models.items()
    }

    return build_optimizers(
        parameter_groups=parameter_groups,
        optimizer_configs=optimizer_configs,
        runtime_context=runtime_context,
    )
