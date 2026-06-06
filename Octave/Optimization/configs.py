#############################
#############################
#############################
# Default optimizer configs
#############################
#############################
#############################


##########################
# SGD Default config
##########################

DEFAULT_SGD_CONFIG = {
    "lr": 1e-3,
    "momentum": 0.0,
    "dampening": 0.0,
    "weight_decay": 0.0,
    "nesterov": False,
    "maximize": False,
    "foreach": None,
    "differentiable": False,
    "fused": None,
}

##########################
# Adam Default config
##########################

DEFAULT_ADAM_CONFIG = {
    "lr": 1e-3,
    "betas": [0.9, 0.999],
    "eps": 1e-8,
    "weight_decay": 0.0,
    "amsgrad": False,
    "maximize": False,
    "foreach": None,
    "capturable": False,
    "differentiable": False,
    "fused": None,
}

##########################
# AdamW Default config
##########################

DEFAULT_ADAMW_CONFIG = {
    "lr": 1e-3,
    "betas": [0.9, 0.999],
    "eps": 1e-8,
    "weight_decay": 1e-2,
    "amsgrad": False,
    "maximize": False,
    "foreach": None,
    "capturable": False,
    "differentiable": False,
    "fused": None,
}

#############################
#############################
#############################
# Default scheduler configs
#############################
#############################
#############################

##########################
# Step LR Default config
##########################

DEFAULT_STEP_LR_CONFIG = {
    "step_size": 10,
    "gamma": 0.1,
}

##############################
# Multistep lr Default config
##############################

DEFAULT_MULTISTEP_LR_CONFIG = {
    "milestones": [30, 80],
    "gamma": 0.1,
}

###############################
# exponential lr Default config
###############################

DEFAULT_EXPONENTIAL_LR_CONFIG = {
    "gamma": 0.95,
}


###################################
# cosine annealing Default config
###################################

DEFAULT_COSINE_ANNEALING_CONFIG = {
    "T_max": 10,
    "eta_min": 0.0,
}


###################################
# reduce on plateau Default config
###################################

DEFAULT_REDUCE_ON_PLATEAU_CONFIG = {
    "mode": "min",
    "factor": 0.1,
    "patience": 10,
}



#############################
#############################
#############################
# Overall optimizer configs
#############################
#############################
#############################


# DEFAULT_OPTIMIZER_CONFIGS = {
#     "sgd": dict(DEFAULT_SGD_CONFIG),
#     "adam": dict(DEFAULT_ADAM_CONFIG),
#     "adamw": dict(DEFAULT_ADAMW_CONFIG),
# }

DEFAULT_OPTIMIZER_CONFIGS = {
    "model": {
        "optimizer_type": "adamw",
        **DEFAULT_ADAMW_CONFIG,
    },
}


#############################
#############################
#############################
# Overall scheduler configs
#############################
#############################
#############################


# DEFAULT_SCHEDULER_CONFIGS = {
#     "step_lr": dict(DEFAULT_STEP_LR_CONFIG),
#     "multistep_lr": dict(DEFAULT_MULTISTEP_LR_CONFIG),
#     "exponential_lr": dict(DEFAULT_EXPONENTIAL_LR_CONFIG),
#     "cosine_annealing": dict(DEFAULT_COSINE_ANNEALING_CONFIG),
#     "reduce_on_plateau": dict(DEFAULT_REDUCE_ON_PLATEAU_CONFIG),
# }

DEFAULT_SCHEDULER_CONFIGS = {
    "model": {
        "scheduler_type": "step_lr",
        **DEFAULT_STEP_LR_CONFIG,
    },
}