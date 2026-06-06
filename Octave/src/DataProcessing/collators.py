import torch

from .base import BaseBatchTransform, BaseCollator


#####################################################
# Default collator
#####################################################


class TSEWaveformCollator(BaseCollator):
    """
    Basic collator for TSE waveform samples.

    Expected sample format:
        {
            "mixture": torch.Tensor,
            "target": torch.Tensor,
            "clue": optional object,
            "metadata": dict,
        }

    Returned batch format:
        {
            "mixture": torch.Tensor,      # (B, ...)
            "target": torch.Tensor,       # (B, ...)
            "clue": list | torch.Tensor | None,
            "metadata": list[dict],
        }

    Batch transforms are applied after collation.
    """

    def __init__(
        self,
        transforms: list[BaseBatchTransform] | None = None,
    ) -> None:
        super().__init__(transforms=transforms)

    def collate_samples(self, samples: list[dict]) -> dict:
        mixtures = [sample["mixture"] for sample in samples]
        targets = [sample["target"] for sample in samples]
        clues = [sample.get("clue", None) for sample in samples]
        metadata = [sample.get("metadata", {}) for sample in samples]

        batch = {
            "mixture": torch.stack(mixtures, dim=0),
            "target": torch.stack(targets, dim=0),
            "clue": self.collate_clues(clues),
            "metadata": metadata,
        }

        return batch

    def collate_clues(self, clues: list) -> object:
        """
        Collate clues conservatively.

        If all clues are tensors with identical shape, stack them.
        If all clues are None, return None.
        Otherwise, keep a list.
        """
        if all(clue is None for clue in clues):
            return None

        if all(isinstance(clue, torch.Tensor) for clue in clues):
            return torch.stack(clues, dim=0)

        return clues