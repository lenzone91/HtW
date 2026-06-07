from typing import Any

import torch
from torch.utils.data import Dataset


class BaseDataset(Dataset):
    """
    Project-level parent class for datasets.

    Expected generic sample convention:
        {
            "input": ...,
            "target": ...,
            "metadata": {...},
        }
    """

    required_sample_keys: tuple[str, ...] = ("input", "target", "metadata")

    def __init__(
        self,
        canonical_dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()

        self.canonical_dtype = canonical_dtype

    
    
    ####################################
    # Usual PyTorch helpers
    ####################################

    def to_tensor(
        self,
        data: Any,
    ) -> torch.Tensor:
        """
        Convert data to a tensor using the dataset canonical dtype.
        """
        tensor_data = torch.tensor(data, dtype=self.canonical_dtype)

        return tensor_data
    


    ####################################
    # Project convention helpers
    ####################################

    @staticmethod
    def build_sample(
        input: Any,
        target: Any,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Build a generic dataset sample.
        """
        if metadata is None:
            metadata = {}

        return {
            "input": input,
            "target": target,
            "metadata": metadata,
        }

    def check_required_keys(
        self,
        sample: dict[str, Any],
    ) -> None:
        """
        Check that a sample contains all required keys.
        """
        missing_keys = []

        for key in self.required_sample_keys:
            if key not in sample:
                missing_keys.append(key)

        if missing_keys:
            raise KeyError(f"Missing sample keys: {missing_keys}")

    def check_sample(
        self,
        sample: dict[str, Any],
    ) -> None:
        """
        Check that a sample follows the generic dataset convention.
        """
        if not isinstance(sample, dict):
            raise TypeError(f"Expected sample to be a dict, got {type(sample)}.")

        self.check_required_keys(sample)

        if not isinstance(sample["metadata"], dict):
            raise TypeError(
                f"Expected sample['metadata'] to be a dict, "
                f"got {type(sample['metadata'])}."
            )