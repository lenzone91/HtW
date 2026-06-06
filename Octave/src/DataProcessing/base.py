from abc import ABC, abstractmethod

import torch

##################################################################
##################################################################
##################################################################
# Base transsform parent class
##################################################################
##################################################################
##################################################################


class BaseBatchTransform(ABC):
    """
    Base class for batch-level transforms.

    A transform maps:

        batch: dict -> batch: dict
    """

    ###################################################################################
    # Usage logic
    ###################################################################################

    def __call__(self, batch: dict) -> dict:
        self.check_input_batch(batch)

        transformed_batch = self.transform(batch)

        self.check_output_batch(transformed_batch)

        return transformed_batch
    
    
    @abstractmethod
    def transform(self, batch: dict) -> dict:
        raise NotImplementedError
    
    #################################################################################
    # Tests logic
    #################################################################################

    def check_input_batch(self, batch: dict) -> None:
        self.check_is_dict(batch, name="input batch")

    def check_output_batch(self, batch: dict) -> None:
        self.check_is_dict(batch, name="output batch")

    def check_is_dict(self, value: object, name: str) -> None:
        if not isinstance(value, dict):
            raise TypeError(
                f"{self.__class__.__name__} expected {name} to be a dict, "
                f"got {type(value).__name__}."
            )

    def check_required_key(self, batch: dict, key: str) -> None:
        if key not in batch:
            raise KeyError(
                f"{self.__class__.__name__} expected key '{key}' in batch."
            )

    def check_is_tensor(self, value: object, name: str) -> None:
        if not isinstance(value, torch.Tensor):
            raise TypeError(
                f"{self.__class__.__name__} expected {name} to be a torch.Tensor, "
                f"got {type(value).__name__}."
            )

    def check_is_batched_tensor(
        self,
        value: object,
        name: str,
        min_ndim: int = 2,
    ) -> None:
        self.check_is_tensor(value, name=name)

        if value.ndim < min_ndim:
            raise ValueError(
                f"{self.__class__.__name__} expected {name} to have at least "
                f"{min_ndim} dimensions, got shape {tuple(value.shape)}."
            )
        

    

    



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

    @abstractmethod
    def collate_samples(self, samples: list[dict]) -> dict:
        """
        Convert a list of sample dictionaries into one batch dictionary.
        """
        raise NotImplementedError

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