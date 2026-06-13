from abc import ABC, abstractmethod


class BaseBatchTransform(ABC):
    """
    Base class for batch-level transforms.
    """

    def __call__(self, batch: dict) -> dict:
        self.check_batch(batch, name="input batch")
        transformed_batch = self.transform(batch)
        self.check_batch(transformed_batch, name="output batch")
        return transformed_batch

    @abstractmethod
    def transform(self, batch: dict) -> dict:
        raise NotImplementedError

    def check_batch(self, batch: dict, name: str) -> None:
        if not isinstance(batch, dict):
            raise TypeError(
                f"{self.__class__.__name__} expected {name} to be a dict, "
                f"got {type(batch).__name__}."
            )


class BaseCollator(ABC):
    """
    Base class for collators.

    A collator maps:
        list[sample dict] -> batch dict
    """

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
        raise NotImplementedError

    def apply_transforms(self, batch: dict) -> dict:
        for transform in self.transforms:
            batch = transform(batch)

        return batch

    def check_samples(self, samples: list[dict]) -> None:
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
        if not isinstance(batch, dict):
            raise TypeError(
                f"{self.__class__.__name__} expected {name} to be a dict, "
                f"got {type(batch).__name__}."
            )
