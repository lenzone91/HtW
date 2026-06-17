import lightning as L
from torch.utils.data import DataLoader

from ..Collators.base import BaseCollator
from ..Datasets.base import BaseDataset


#############################################
# Base data module
#############################################


class BaseDataModule(L.LightningDataModule):
    """
    Generic LightningDataModule for already-built datasets and collators.

    Responsibilities:
        - store one dataset per phase (train/val/test);
        - store one collator per phase;
        - store one DataLoader config per phase;
        - build phase-specific DataLoaders.

    It does not:
        - build datasets or collators (the factory does, via sub-builds);
        - perform preprocessing;
        - know model/objective logic.
    """

    def __init__(
        self,
        datasets: dict[str, BaseDataset | None],
        collators: dict[str, BaseCollator | None],
        dataloader_configs: dict[str, dict],
    ) -> None:
        super().__init__()

        self.datasets = datasets
        self.collators = collators
        self.dataloader_configs = dataloader_configs

        self.check_phase_consistency()

    def train_dataloader(self) -> DataLoader | None:
        return self.make_dataloader("train")

    def val_dataloader(self) -> DataLoader | None:
        return self.make_dataloader("val")

    def test_dataloader(self) -> DataLoader | None:
        return self.make_dataloader("test")

    def make_dataloader(self, phase: str) -> DataLoader | None:
        self.check_known_phase(phase)

        dataset = self.datasets.get(phase)

        if dataset is None:
            return None

        collator = self.collators.get(phase)
        dataloader_config = self.dataloader_configs.get(phase, {})

        if collator is None:
            return DataLoader(dataset=dataset, **dataloader_config)

        return DataLoader(
            dataset=dataset,
            collate_fn=collator,
            **dataloader_config,
        )

    #############################################
    # Validation helpers
    #############################################

    def check_phase_consistency(self) -> None:
        for phase, dataset in self.datasets.items():
            if dataset is None:
                continue

            if phase not in self.collators:
                raise KeyError(f"Missing collator for phase '{phase}'.")

            if phase not in self.dataloader_configs:
                raise KeyError(f"Missing DataLoader config for phase '{phase}'.")

    def check_known_phase(self, phase: str) -> None:
        if phase not in self.datasets:
            raise KeyError(
                f"Unknown phase '{phase}'. "
                f"Available phases are: {sorted(self.datasets.keys())}."
            )
