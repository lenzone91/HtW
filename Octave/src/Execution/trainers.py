from lightning.pytorch import Trainer

from .configs import DEFAULT_LIGHTNING_TRAINER_CONFIG
from .registry import TRAINER_REGISTRY


@TRAINER_REGISTRY.register_class(
    name="lightning",
    default_config=DEFAULT_LIGHTNING_TRAINER_CONFIG,
    type_field="trainer_type",
)
class OctaveTrainer(Trainer):
    """
    Registered Lightning Trainer adapter.

    It only exists to expose Trainer construction through the shared
    Workflow/Factory registry logic.
    """

    pass