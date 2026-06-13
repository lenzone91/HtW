import random

import numpy as np
import torch


def setup_reproducibility(config: dict | None = None) -> dict:
    config = dict(config or {})

    seed = config.get("seed", 42)
    deterministic = config.get("deterministic", False)
    cudnn_benchmark = config.get("cudnn_benchmark", False)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = cudnn_benchmark
    torch.use_deterministic_algorithms(deterministic)

    return {
        "seed": seed,
        "deterministic": deterministic,
        "cudnn_benchmark": cudnn_benchmark,
    }
