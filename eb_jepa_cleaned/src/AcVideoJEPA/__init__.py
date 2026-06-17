"""
AcVideoJEPA: the concrete action-conditioned video JEPA experiment (it also owns
the JEPA objective; Decision 33). Importing this package registers all the
experiment concretes onto the AIML registries: the backbone models, the latent
rollout, the JEPA objective metrics, the AcVideoJepaModule, and the two-rooms
dataset + collator. The runnable Hydra config tree lives in `configs/`.
"""

from .Data import Collators as _collators  # noqa: F401  (registers collator)
from .Data import Datasets as _datasets  # noqa: F401  (registers dataset)
from .Models import Modules as _modules  # noqa: F401  (registers module + deps)
