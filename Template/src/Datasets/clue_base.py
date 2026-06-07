from typing import Any

from .base import BaseDataset


class BaseClueDataset(BaseDataset):
    """
    Base class for clue datasets.

    Expected clue sample convention:
        {
            "clue": ...,
            "metadata": {...},
        }

    A clue dataset maps a lookup key to a clue sample.

    This class does not:
        - extract clues;
        - define TSE pairing logic;
        - assume whether keys are speaker ids, sample ids, or enrollment ids.
    """

    required_sample_keys: tuple[str, ...] = ("clue", "metadata")

    @staticmethod
    def build_clue_sample(
        clue: Any,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Build a clue sample.
        """
        if metadata is None:
            metadata = {}
        return {
            "clue": clue,
            "metadata": metadata,
        }

    def check_sample(self, sample: dict[str, Any]) -> None:
        """
        Check that a clue sample follows the clue dataset convention.
        """
        if not isinstance(sample, dict):
            raise TypeError(f"Expected clue sample to be a dict, got {type(sample)}.")

        self.check_required_keys(sample)

        if not isinstance(sample["metadata"], dict):
            raise TypeError(
                f"Expected clue sample['metadata'] to be a dict, "
                f"got {type(sample['metadata'])}."
            )

    def check_key(self, key: Any) -> None:
        """
        Check that a clue lookup key is valid.

        Concrete clue datasets may override this method.
        """
        if key is None:
            raise ValueError("Clue lookup key cannot be None.")