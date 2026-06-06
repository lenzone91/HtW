
from .base import BaseDataset
from .clue_base import BaseClueDataset

from typing import Any


#####################################
# General TSE dataset
##################################### 

class BaseTSEDataset(BaseDataset):
    """
    Base class for Target Speech Extraction datasets.

    Expected TSE sample convention:
        {
            "mixture": ...,
            "target": ...,
            "clue": ...,        # may be None
            "metadata": {...},
        }

    This class only defines the TSE-level sample convention.
    """

    required_sample_keys: tuple[str, ...] = (
        "mixture",
        "target",
        "clue",
        "metadata",
    )

    @staticmethod
    def build_tse_sample(
        mixture: Any,
        target: Any,
        clue: Any | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Build a TSE sample.
        """
        if metadata is None:
            metadata = {}

        return {
            "mixture": mixture,
            "target": target,
            "clue": clue,
            "metadata": metadata,
        }
    


##########################################
# TSE dataset when a clue is used for sure
##########################################

class BaseTSEDatasetWithClue(BaseTSEDataset):
    """
    Base class for TSE datasets where a clue is always expected.

    This class is useful for datasets or wrappers that guarantee that
    each returned sample contains an actual clue.
    """

    @staticmethod
    def build_tse_sample(
        mixture: Any,
        target: Any,
        clue: Any,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Build a clued TSE sample.
        """
        if clue is None:
            raise ValueError("A clued TSE sample cannot have clue=None.")

        if metadata is None:
            metadata = {}

        return {
            "mixture": mixture,
            "target": target,
            "clue": clue,
            "metadata": metadata,
        }
    


##################################################
# General TSE dataset with a clue and a pairing
##################################################



class PairedTSEDatasetWithClue(BaseTSEDatasetWithClue):
    """
    TSE dataset built by pairing a base dataset with a clue dataset.

    Expected base dataset sample convention:
        {
            "input": mixture,
            "target": target,
            "metadata": {...},
        }

    Expected clue dataset sample convention:
        {
            "clue": clue,
            "metadata": {...},
        }

    Returned TSE sample convention:
        {
            "mixture": mixture,
            "target": target,
            "clue": clue,
            "metadata": {
                ...base_metadata,
                "clue_metadata": {...},
            },
        }

    The pairing is done through a key stored in the base sample metadata.
    """

    def __init__(
        self,
        base_dataset: BaseDataset,
        clue_dataset: BaseClueDataset,
        clue_lookup_key: str,
    ) -> None:
        super().__init__()

        self.base_dataset = base_dataset
        self.clue_dataset = clue_dataset
        self.clue_lookup_key = clue_lookup_key

    def __len__(self) -> int:
        """
        Return the number of base samples.
        """
        return len(self.base_dataset)

    def __getitem__(self, index: int) -> dict[str, Any]:
        """
        Return a TSE sample with a guaranteed clue.
        """
        base_sample = self.base_dataset[index]

        mixture = base_sample["input"]
        target = base_sample["target"]
        base_metadata = base_sample["metadata"]

        clue_key = self.get_clue_key(base_metadata)
        clue_sample = self.clue_dataset[clue_key]

        clue = clue_sample["clue"]
        clue_metadata = clue_sample["metadata"]

        metadata = self.build_metadata(
            base_metadata=base_metadata,
            clue_metadata=clue_metadata,
        )

        return self.build_tse_sample(
            mixture=mixture,
            target=target,
            clue=clue,
            metadata=metadata,
        )

    def get_clue_key(self, base_metadata: dict[str, Any]) -> Any:
        """
        Extract the clue lookup key from the base sample metadata.
        """
        if self.clue_lookup_key not in base_metadata:
            raise KeyError(
                f"Missing clue lookup key '{self.clue_lookup_key}' "
                f"in base sample metadata."
            )

        return base_metadata[self.clue_lookup_key]

    @staticmethod
    def build_metadata(
        base_metadata: dict[str, Any],
        clue_metadata: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Merge base metadata and clue metadata without key collisions.
        """
        metadata = dict(base_metadata)
        metadata["clue_metadata"] = clue_metadata

        return metadata