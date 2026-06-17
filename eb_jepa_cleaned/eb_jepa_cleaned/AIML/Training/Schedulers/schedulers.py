import torch

from .registry import SCHEDULER_REGISTRY


#############################################
# Schedulers (thin torch LR-scheduler wrappers)
#############################################

# Each object is preceded by its own default config.


DEFAULT_STEP_LR_CONFIG = {"step_size": 10, "gamma": 0.1}


@SCHEDULER_REGISTRY.register_class(
    name="step_lr",
    default_config={"scheduler_type": "step_lr", **DEFAULT_STEP_LR_CONFIG},
    type_field="scheduler_type",
)
class StepLR(torch.optim.lr_scheduler.StepLR):
    pass


DEFAULT_MULTISTEP_LR_CONFIG = {"milestones": [30, 80], "gamma": 0.1}


@SCHEDULER_REGISTRY.register_class(
    name="multistep_lr",
    default_config={"scheduler_type": "multistep_lr", **DEFAULT_MULTISTEP_LR_CONFIG},
    type_field="scheduler_type",
)
class MultiStepLR(torch.optim.lr_scheduler.MultiStepLR):
    pass


DEFAULT_EXPONENTIAL_LR_CONFIG = {"gamma": 0.95}


@SCHEDULER_REGISTRY.register_class(
    name="exponential_lr",
    default_config={
        "scheduler_type": "exponential_lr",
        **DEFAULT_EXPONENTIAL_LR_CONFIG,
    },
    type_field="scheduler_type",
)
class ExponentialLR(torch.optim.lr_scheduler.ExponentialLR):
    pass


DEFAULT_COSINE_ANNEALING_CONFIG = {"T_max": 10, "eta_min": 0.0}


@SCHEDULER_REGISTRY.register_class(
    name="cosine_annealing",
    default_config={
        "scheduler_type": "cosine_annealing",
        **DEFAULT_COSINE_ANNEALING_CONFIG,
    },
    type_field="scheduler_type",
)
class CosineAnnealingLR(torch.optim.lr_scheduler.CosineAnnealingLR):
    pass


DEFAULT_REDUCE_ON_PLATEAU_CONFIG = {"mode": "min", "factor": 0.1, "patience": 10}


@SCHEDULER_REGISTRY.register_class(
    name="reduce_on_plateau",
    default_config={
        "scheduler_type": "reduce_on_plateau",
        **DEFAULT_REDUCE_ON_PLATEAU_CONFIG,
    },
    type_field="scheduler_type",
)
class ReduceLROnPlateau(torch.optim.lr_scheduler.ReduceLROnPlateau):
    pass
