import torch

from .base import BaseBatchTransform, BaseCollator
from ..Datasets.two_rooms import AC_VIDEO_JEPA_SAMPLE_KEYS

from .configs import DEFAULT_AC_VIDEO_JEPA_COLLATOR_CONFIG
from .registry import COLLATOR_REGISTRY
from ...Workflow.Factory.registry import FieldResolution


AC_VIDEO_JEPA_TENSOR_KEYS = (
    "states",
    "actions",
    "locations",
    "wall_x",
    "door_y",
)



@COLLATOR_REGISTRY.register_class(
    name="ac_video_jepa",
    default_config=DEFAULT_AC_VIDEO_JEPA_COLLATOR_CONFIG,
)
class AcVideoJepaCollator(BaseCollator):
    """
    Collate AcVideoJepa semantic samples into semantic batch dictionaries.
    """

    def __init__(
        self,
        transforms: list[BaseBatchTransform] | None = None,
    ) -> None:
        super().__init__(transforms=transforms)

    def collate_samples(self, samples: list[dict]) -> dict:
        self.check_required_sample_keys(samples)
        self.check_tensor_values(samples)

        batch = {
            key: torch.stack([sample[key] for sample in samples], dim=0)
            for key in AC_VIDEO_JEPA_TENSOR_KEYS
        }
        batch["metadata"] = [sample["metadata"] for sample in samples]

        return batch

    def check_required_sample_keys(self, samples: list[dict]) -> None:
        for sample_index, sample in enumerate(samples):
            missing_keys = [
                key
                for key in AC_VIDEO_JEPA_SAMPLE_KEYS
                if key not in sample
            ]

            if missing_keys:
                raise KeyError(
                    f"Sample {sample_index} is missing AcVideoJepa keys: "
                    f"{missing_keys}."
                )

    def check_tensor_values(self, samples: list[dict]) -> None:
        for sample_index, sample in enumerate(samples):
            for key in AC_VIDEO_JEPA_TENSOR_KEYS:
                if not isinstance(sample[key], torch.Tensor):
                    raise TypeError(
                        f"Sample {sample_index} key '{key}' must be a torch.Tensor, "
                        f"got {type(sample[key]).__name__}."
                    )
