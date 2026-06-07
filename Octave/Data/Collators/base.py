from abc import ABC, abstractmethod

import torch

from torch.utils.data import default_collate

from ..DataTransforms.base import BaseBatchTransform
    



##################################################################
##################################################################
##################################################################
# Base collator parent class
##################################################################
##################################################################
##################################################################

class BaseCollator(ABC):
    """
    Base class for collators.

    A collator maps:

        samples: list[dict] -> batch: dict

    Pipeline:
        1. validate input samples;
        2. collate samples into a batch dictionary;
        3. apply batch transforms in order;
        4. return the processed batch.

    Transforms may mutate the batch dictionary in-place.
    """

    #########################################################
    # Usage logic
    #########################################################

    def __init__(self, transforms: list[BaseBatchTransform] | None = None) -> None:
        self.transforms = transforms or []

    def __call__(self, samples: list[dict]) -> dict:
        self.check_samples(samples)

        batch = self.collate_samples(samples)
        self.check_batch(batch, name="collated batch")

        batch = self.apply_transforms(batch)
        self.check_batch(batch, name="transformed batch")

        return batch

    def collate_samples(self, samples: list[dict]) -> dict:
        """
        Convert a list of sample dictionaries into one batch dictionary.

        This uses PyTorch's default collation logic:
        tensors are stacked, numerical values are tensorized, and standard
        containers are collated recursively.
        """
        return default_collate(samples)

    def apply_transforms(self, batch: dict) -> dict:
        """
        Apply all batch transforms in their registered order.
        """
        for transform in self.transforms:
            batch = transform(batch)

        return batch
    

    #########################################################################################
    # Test logic
    #########################################################################################

    def check_samples(self, samples: list[dict]) -> None:
        """
        Check that the collator receives a non-empty list of sample dictionaries.
        """
        if not isinstance(samples, list):
            raise TypeError(
                f"{self.__class__.__name__} expected samples to be a list, "
                f"got {type(samples).__name__}."
            )

        if len(samples) == 0:
            raise ValueError(f"{self.__class__.__name__} received an empty sample list.")

        for sample_index, sample in enumerate(samples):
            if not isinstance(sample, dict):
                raise TypeError(
                    f"{self.__class__.__name__} expected each sample to be a dict, "
                    f"but sample {sample_index} has type {type(sample).__name__}."
                )

    def check_batch(self, batch: dict, name: str) -> None:
        """
        Check that a produced batch is a dictionary.
        """
        if not isinstance(batch, dict):
            raise TypeError(
                f"{self.__class__.__name__} expected {name} to be a dict, "
                f"got {type(batch).__name__}."
            )