import torch

from ..Workflow.Factory.base import BaseBuilder, BaseBuilderDispatcher
from .configs import (
    DEFAULT_SGD_CONFIG,
    DEFAULT_ADAM_CONFIG,
    DEFAULT_ADAMW_CONFIG,
    DEFAULT_STEP_LR_CONFIG,
    DEFAULT_MULTISTEP_LR_CONFIG,
    DEFAULT_EXPONENTIAL_LR_CONFIG,
    DEFAULT_COSINE_ANNEALING_CONFIG,
    DEFAULT_REDUCE_ON_PLATEAU_CONFIG,
)


"""
Optimization factory utilities.

Optimizers and schedulers are instantiated inside
LightningModule.configure_optimizers.

This file only converts plain serializable configuration dictionaries into
PyTorch optimizer / scheduler objects when configure_optimizers is called.
"""


##############################
# Optimizer builder
##############################


class OptimizerBuilder(BaseBuilder):
    def __init__(
        self,
        optimizer_class,
        default_config: dict,
        strict: bool = True,
    ):
        super().__init__(
            default_config=default_config,
            strict=strict,
            check_default_keys=True,
        )

        self.optimizer_class = optimizer_class

    def build_from_config(
        self,
        config: dict,
        runtime_context: dict | None = None,
        parameters=None,
    ):
        return self.optimizer_class(parameters, **config)

    def prepare_config(self, config: dict) -> dict:
        config = super().prepare_config(config)

        # Used only by the dispatcher for routing.
        # Not a torch optimizer argument.
        config.pop("optimizer_type", None)

        return config


########################
# Optimizer registry
########################


OPTIMIZER_BUILDERS_REGISTRY = {
    "sgd": OptimizerBuilder(torch.optim.SGD, DEFAULT_SGD_CONFIG),
    "adam": OptimizerBuilder(torch.optim.Adam, DEFAULT_ADAM_CONFIG),
    "adamw": OptimizerBuilder(torch.optim.AdamW, DEFAULT_ADAMW_CONFIG),
}


######################
# Optimizer dispatcher
######################


class OptimizerBuilderDispatcher(BaseBuilderDispatcher):
    """
    Build optimizers from named optimizer configs.

    Expected format:
        optimizer_configs[optimizer_name] = {
            "optimizer_type": "sgd" | "adam" | "adamw",
            ...
        }

    Extra build context:
        parameter_groups[optimizer_name] = parameters
    """

    def __init__(
        self,
        builder_registry: dict = OPTIMIZER_BUILDERS_REGISTRY,
        strict: bool = True,
    ):
        super().__init__(
            default_config={},
            builder_registry=builder_registry,
            strict=strict,
            check_default_keys=False,
        )

    def check_config(
        self,
        config: dict,
        runtime_context: dict | None = None,
        parameter_groups: dict | None = None,
    ) -> bool:
        # We intentionally do not call BaseBuilderDispatcher.check_config.
        # It assumes object_name is the registry key, but here the registry key
        # is optimizer_config["optimizer_type"].
        if not BaseBuilder.check_config(
            self,
            config=config,
            runtime_context=runtime_context,
        ):
            return False

        if not isinstance(parameter_groups, dict):
            self.handle_error(
                f"parameter_groups must be a dictionary, got {type(parameter_groups)}."
            )
            return False

        for optimizer_name, optimizer_config in config.items():
            if not self.check_object_config(optimizer_name, optimizer_config):
                return False

            if not self.check_optimizer_type_is_registered(optimizer_config):
                return False

            if not self.check_parameter_group_exists(
                optimizer_name=optimizer_name,
                parameter_groups=parameter_groups,
            ):
                return False

        return True

    def build_from_config(
        self,
        config: dict,
        runtime_context: dict | None = None,
        parameter_groups: dict | None = None,
    ) -> dict[str, torch.optim.Optimizer]:
        optimizers = {}

        for optimizer_name, optimizer_config in config.items():
            optimizers[optimizer_name] = self.build_one(
                optimizer_name=optimizer_name,
                optimizer_config=optimizer_config,
                runtime_context=runtime_context,
                parameter_groups=parameter_groups,
            )

        return optimizers

    def build_one(
        self,
        optimizer_name: str,
        optimizer_config: dict,
        runtime_context: dict | None = None,
        parameter_groups: dict | None = None,
    ) -> torch.optim.Optimizer:
        optimizer_type = optimizer_config["optimizer_type"]
        builder = self.builder_registry[optimizer_type]

        return builder(
            config=optimizer_config,
            runtime_context=runtime_context,
            parameters=parameter_groups[optimizer_name],
        )

    def check_optimizer_type_is_registered(
        self,
        optimizer_config: dict,
    ) -> bool:
        optimizer_type = optimizer_config.get("optimizer_type")

        if optimizer_type in self.builder_registry:
            return True

        self.handle_error(
            f"Unknown optimizer type '{optimizer_type}'. "
            f"Available optimizer types are: {sorted(self.builder_registry.keys())}."
        )
        return False

    def check_parameter_group_exists(
        self,
        optimizer_name: str,
        parameter_groups: dict,
    ) -> bool:
        if optimizer_name in parameter_groups:
            return True

        self.handle_error(
            f"No parameter group found for optimizer '{optimizer_name}'. "
            f"Available parameter groups are: {sorted(parameter_groups.keys())}."
        )
        return False


##############################
# Scheduler builder
##############################


class SchedulerBuilder(BaseBuilder):
    def __init__(
        self,
        scheduler_class,
        default_config: dict,
        strict: bool = True,
    ):
        super().__init__(
            default_config=default_config,
            strict=strict,
            check_default_keys=True,
        )

        self.scheduler_class = scheduler_class

    def build_from_config(
        self,
        config: dict,
        runtime_context: dict | None = None,
        optimizer: torch.optim.Optimizer | None = None,
    ):
        return self.scheduler_class(optimizer, **config)

    def prepare_config(self, config: dict) -> dict:
        config = super().prepare_config(config)

        # Used only by the dispatcher for routing.
        # Not a torch scheduler argument.
        config.pop("scheduler_type", None)

        return config


#########################
# Scheduler registry
#########################


SCHEDULER_BUILDERS_REGISTRY = {
    "step_lr": SchedulerBuilder(
        torch.optim.lr_scheduler.StepLR,
        DEFAULT_STEP_LR_CONFIG,
    ),

    "multistep_lr": SchedulerBuilder(
        torch.optim.lr_scheduler.MultiStepLR,
        DEFAULT_MULTISTEP_LR_CONFIG,
    ),

    "exponential_lr": SchedulerBuilder(
        torch.optim.lr_scheduler.ExponentialLR,
        DEFAULT_EXPONENTIAL_LR_CONFIG,
    ),

    "cosine_annealing": SchedulerBuilder(
        torch.optim.lr_scheduler.CosineAnnealingLR,
        DEFAULT_COSINE_ANNEALING_CONFIG,
    ),

    "reduce_on_plateau": SchedulerBuilder(
        torch.optim.lr_scheduler.ReduceLROnPlateau,
        DEFAULT_REDUCE_ON_PLATEAU_CONFIG,
    ),
}


#########################
# Scheduler dispatcher
#########################


class SchedulerBuilderDispatcher(BaseBuilderDispatcher):
    """
    Build schedulers from named scheduler configs.

    Expected format:
        scheduler_configs[scheduler_name] = {
            "scheduler_type": "step_lr" | "multistep_lr" | ...
            ...
        }

    Extra build context:
        optimizer_groups[scheduler_name] = optimizer
    """

    def __init__(
        self,
        builder_registry: dict = SCHEDULER_BUILDERS_REGISTRY,
        strict: bool = True,
    ):
        super().__init__(
            default_config={},
            builder_registry=builder_registry,
            strict=strict,
            check_default_keys=False,
        )

    def check_config(
        self,
        config: dict,
        runtime_context: dict | None = None,
        optimizer_groups: dict | None = None,
    ) -> bool:
        if not BaseBuilder.check_config(
            self,
            config=config,
            runtime_context=runtime_context,
        ):
            return False

        if not isinstance(optimizer_groups, dict):
            self.handle_error(
                f"optimizer_groups must be a dictionary, got {type(optimizer_groups)}."
            )
            return False

        for scheduler_name, scheduler_config in config.items():
            if not self.check_object_config(scheduler_name, scheduler_config):
                return False

            if not self.check_scheduler_type_is_registered(scheduler_config):
                return False

            if not self.check_optimizer_group_exists(
                scheduler_name=scheduler_name,
                optimizer_groups=optimizer_groups,
            ):
                return False

        return True

    def build_from_config(
        self,
        config: dict,
        runtime_context: dict | None = None,
        optimizer_groups: dict | None = None,
    ) -> dict:
        schedulers = {}

        for scheduler_name, scheduler_config in config.items():
            schedulers[scheduler_name] = self.build_one(
                scheduler_name=scheduler_name,
                scheduler_config=scheduler_config,
                runtime_context=runtime_context,
                optimizer_groups=optimizer_groups,
            )

        return schedulers

    def build_one(
        self,
        scheduler_name: str,
        scheduler_config: dict,
        runtime_context: dict | None = None,
        optimizer_groups: dict | None = None,
    ):
        scheduler_type = scheduler_config["scheduler_type"]
        builder = self.builder_registry[scheduler_type]

        return builder(
            config=scheduler_config,
            runtime_context=runtime_context,
            optimizer=optimizer_groups[scheduler_name],
        )

    def check_scheduler_type_is_registered(
        self,
        scheduler_config: dict,
    ) -> bool:
        scheduler_type = scheduler_config.get("scheduler_type")

        if scheduler_type in self.builder_registry:
            return True

        self.handle_error(
            f"Unknown scheduler type '{scheduler_type}'. "
            f"Available scheduler types are: {sorted(self.builder_registry.keys())}."
        )
        return False

    def check_optimizer_group_exists(
        self,
        scheduler_name: str,
        optimizer_groups: dict,
    ) -> bool:
        if scheduler_name in optimizer_groups:
            return True

        self.handle_error(
            f"No optimizer found for scheduler '{scheduler_name}'. "
            f"Available optimizers are: {sorted(optimizer_groups.keys())}."
        )
        return False


#################################
# Wrappers
#################################


def build_optimizers(
    parameter_groups: dict,
    optimizer_configs: dict,
    runtime_context: dict | None = None,
    strict: bool = True,
) -> dict[str, torch.optim.Optimizer]:
    dispatcher = OptimizerBuilderDispatcher(strict=strict)

    return dispatcher(
        config=optimizer_configs,
        runtime_context=runtime_context,
        parameter_groups=parameter_groups,
    )


def build_optimizers_from_models(
    models: dict[str, torch.nn.Module],
    optimizer_configs: dict,
    runtime_context: dict | None = None,
    strict: bool = True,
) -> dict[str, torch.optim.Optimizer]:
    parameter_groups = {
        model_name: model.parameters()
        for model_name, model in models.items()
    }

    return build_optimizers(
        parameter_groups=parameter_groups,
        optimizer_configs=optimizer_configs,
        runtime_context=runtime_context,
        strict=strict,
    )


def build_schedulers(
    optimizer_groups: dict[str, torch.optim.Optimizer],
    scheduler_configs: dict,
    runtime_context: dict | None = None,
    strict: bool = True,
) -> dict:
    dispatcher = SchedulerBuilderDispatcher(strict=strict)

    return dispatcher(
        config=scheduler_configs,
        runtime_context=runtime_context,
        optimizer_groups=optimizer_groups,
    )




def build_optimizer(
    parameters,
    optimizer_config: dict,
    runtime_context: dict | None = None,
    strict: bool = True,
) -> torch.optim.Optimizer:
    """
    Build one optimizer from one optimizer config.

    Expected optimizer_config:
        {
            "optimizer_type": "sgd" | "adam" | "adamw",
            ...
        }
    """
    optimizer_configs = {
        "model": optimizer_config,
    }

    parameter_groups = {
        "model": parameters,
    }

    optimizers = build_optimizers(
        parameter_groups=parameter_groups,
        optimizer_configs=optimizer_configs,
        runtime_context=runtime_context,
        strict=strict,
    )

    return optimizers["model"]


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    scheduler_config: dict,
    runtime_context: dict | None = None,
    strict: bool = True,
):
    """
    Build one scheduler from one scheduler config.

    Expected scheduler_config:
        {
            "scheduler_type": "step_lr" | "multistep_lr" | ...
            ...
        }
    """
    scheduler_configs = {
        "model": scheduler_config,
    }

    optimizer_groups = {
        "model": optimizer,
    }

    schedulers = build_schedulers(
        optimizer_groups=optimizer_groups,
        scheduler_configs=scheduler_configs,
        runtime_context=runtime_context,
        strict=strict,
    )

    return schedulers["model"]