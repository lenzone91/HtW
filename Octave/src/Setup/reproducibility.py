import random

import numpy as np
import torch


#############################################
# Reproducibility setup
#############################################

# This file centralizes reproducibility-related setup.
#
# It handles runtime side effects such as seed fixing and backend flags.
# It does not build models, dataloaders, trainers, or callbacks.


def setup_reproducibility(reproducibility_config: dict | None = None) -> dict:
    """
    Apply reproducibility-related setup and return reproducibility context.
    """
    if reproducibility_config is None:
        reproducibility_config = {}

    seed = reproducibility_config.get("seed", None)
    deterministic = reproducibility_config.get("deterministic", False)
    cudnn_benchmark = reproducibility_config.get("cudnn_benchmark", False)

    if seed is not None:
        set_global_seed(seed)

    set_deterministic_mode(deterministic)
    set_cudnn_benchmark(cudnn_benchmark)

    return {
        "seed": seed,
        "deterministic": deterministic,
        "cudnn_benchmark": cudnn_benchmark,
    }


#############################################
# Seed setup
#############################################

# Seed fixing improves repeatability.
# Strict reproducibility also depends on hardware, backend operations,
# dataloader behavior, package versions, and deterministic settings.


def set_global_seed(seed: int) -> None:
    """
    Set the main random seeds used in the project.
    """
    check_seed(seed)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def check_seed(seed: int) -> None:
    """
    Check that the provided seed is a valid integer seed.
    """
    if not isinstance(seed, int):
        raise TypeError(f"seed must be an int, got {type(seed)}.")

    if seed < 0:
        raise ValueError(f"seed must be non-negative, got {seed}.")


#############################################
# Backend setup
#############################################

# Backend flags control part of PyTorch numerical reproducibility.
# They should be explicit because they may affect performance.


def set_deterministic_mode(deterministic: bool) -> None:
    """
    Enable or disable deterministic PyTorch algorithms.
    """
    if not isinstance(deterministic, bool):
        raise TypeError(
            f"deterministic must be a bool, got {type(deterministic)}."
        )

    torch.use_deterministic_algorithms(deterministic)


def set_cudnn_benchmark(cudnn_benchmark: bool) -> None:
    """
    Enable or disable cuDNN benchmarking.
    """
    if not isinstance(cudnn_benchmark, bool):
        raise TypeError(
            f"cudnn_benchmark must be a bool, got {type(cudnn_benchmark)}."
        )

    torch.backends.cudnn.benchmark = cudnn_benchmark