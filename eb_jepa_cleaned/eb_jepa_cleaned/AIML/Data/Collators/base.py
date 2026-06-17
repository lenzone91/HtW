from abc import ABC, abstractmethod

from ..BatchTransform.base import BaseBatchTransform


#############################################
# Base collator
#############################################


class BaseCollator(ABC):
    """
    Base class for collators.

    A collator maps a list of samples to a batch:

        samples: list[dict] -> batch: dict

    Pipeline:
        1. validate input samples;
        2. collate samples into a batch dictionary;
        3. apply batch transforms in order;
        4. return the processed batch.

    The collator is family-agnostic: `batch_transforms` is an ordered list of
    `BaseBatchTransform`, which may contain augmentations and/or adaptations. How
    a concrete collator assembles that list (and from which families) is its own
    concern. This is an abstract base: no default config, no registration.
    """

    def __init__(
        self,
        batch_transforms: list[BaseBatchTransform]
        | dict[str, BaseBatchTransform]
        | None = None,
    ) -> None:
        self.batch_transforms = self.prepare_batch_transforms(batch_transforms)

    #############################################
    # Usage
    #############################################

    def __call__(self, samples: list[dict]) -> dict:
        self.check_samples(samples)

        batch = self.collate_samples(samples)
        self.check_batch(batch, name="collated batch")

        batch = self.apply_batch_transforms(batch)
        self.check_batch(batch, name="transformed batch")

        return batch

    @abstractmethod
    def collate_samples(self, samples: list[dict]) -> dict:
        """
        Convert a list of sample dictionaries into one batch dictionary.
        """
        raise NotImplementedError

    def prepare_batch_transforms(
        self,
        batch_transforms: list[BaseBatchTransform]
        | dict[str, BaseBatchTransform]
        | None,
    ) -> list[BaseBatchTransform]:
        """
        Normalize a transform container to an ordered list.
        """
        if batch_transforms is None:
            return []

        if isinstance(batch_transforms, dict):
            return list(batch_transforms.values())

        return batch_transforms

    def apply_batch_transforms(self, batch: dict) -> dict:
        """
        Apply all batch transforms in order.
        """
        for transform in self.batch_transforms:
            batch = transform(batch)

        return batch

    #############################################
    # Validation helpers
    #############################################

    def check_samples(self, samples: list[dict]) -> None:
        """
        Check that the collator receives a non-empty list of sample dicts.
        """
        if not isinstance(samples, list):
            raise TypeError(
                f"{self.__class__.__name__} expected samples to be a list, "
                f"got {type(samples).__name__}."
            )

        if len(samples) == 0:
            raise ValueError(
                f"{self.__class__.__name__} received an empty sample list."
            )

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
