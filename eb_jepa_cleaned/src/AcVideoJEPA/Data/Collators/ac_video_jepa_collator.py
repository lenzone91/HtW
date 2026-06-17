"""
AcVideoJEPA collator.

Collates the semantic two-rooms samples into a batch dictionary: stacks the
tensor fields and gathers metadata. The module reads `states` / `actions`.
"""

import torch

from ....AIML.Data.Collators.base import BaseCollator
from ....AIML.Data.Collators.registry import COLLATOR_REGISTRY
from ..Datasets.two_rooms_dataset import AC_VIDEO_JEPA_SAMPLE_KEYS

# Tensor fields stacked into the batch (everything but `metadata`).
AC_VIDEO_JEPA_TENSOR_KEYS = tuple(k for k in AC_VIDEO_JEPA_SAMPLE_KEYS if k != "metadata")

DEFAULT_AC_VIDEO_JEPA_COLLATOR_CONFIG = {"collator_type": "ac_video_jepa"}


@COLLATOR_REGISTRY.register_class(
    name="ac_video_jepa",
    default_config=DEFAULT_AC_VIDEO_JEPA_COLLATOR_CONFIG,
    type_field="collator_type",
)
class AcVideoJepaCollator(BaseCollator):
    """Collate semantic two-rooms samples into a semantic batch dictionary."""

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
        for index, sample in enumerate(samples):
            missing = [k for k in AC_VIDEO_JEPA_SAMPLE_KEYS if k not in sample]
            if missing:
                raise KeyError(f"Sample {index} is missing keys: {missing}.")

    def check_tensor_values(self, samples: list[dict]) -> None:
        for index, sample in enumerate(samples):
            for key in AC_VIDEO_JEPA_TENSOR_KEYS:
                if not isinstance(sample[key], torch.Tensor):
                    raise TypeError(
                        f"Sample {index} key '{key}' must be a torch.Tensor, "
                        f"got {type(sample[key]).__name__}."
                    )
