import lightning as L
from torch.utils.data import DataLoader, Dataset


class BaseDataModule(L.LightningDataModule):
    """
    Generic LightningDataModule for already-built datasets and collators.
    """

    def __init__(
        self,
        datasets: dict[str, Dataset | None],
        collators: dict[str, object | None],
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
        return self.make_optional_dataloader("val")

    def test_dataloader(self) -> DataLoader | None:
        return self.make_optional_dataloader("test")

    def make_dataloader(self, phase: str) -> DataLoader | None:
        self.check_known_phase(phase)

        dataset = self.datasets.get(phase)

        if dataset is None:
            return None

        return DataLoader(
            dataset=dataset,
            collate_fn=self.collators[phase],
            **self.dataloader_configs[phase],
        )

    def make_optional_dataloader(self, phase: str):
        dataloader = self.make_dataloader(phase)

        if dataloader is None:
            return []

        return dataloader

    def check_phase_consistency(self) -> None:
        for phase, dataset in self.datasets.items():
            if dataset is None:
                continue

            if phase not in self.collators:
                raise KeyError(f"Missing collator for phase '{phase}'.")

            if self.collators[phase] is None:
                raise ValueError(f"Collator for phase '{phase}' is None.")

            if phase not in self.dataloader_configs:
                raise KeyError(f"Missing DataLoader config for phase '{phase}'.")

            self.check_dataloader_config(
                phase=phase,
                dataloader_config=self.dataloader_configs[phase],
            )

    def check_dataloader_config(
        self,
        phase: str,
        dataloader_config: dict,
    ) -> None:
        if not isinstance(dataloader_config, dict):
            raise TypeError(
                f"DataLoader config for phase '{phase}' must be a dictionary, "
                f"got {type(dataloader_config).__name__}."
            )

        num_workers = dataloader_config.get("num_workers", 0)
        persistent_workers = dataloader_config.get("persistent_workers", False)

        if persistent_workers and num_workers <= 0:
            raise ValueError(
                "Invalid DataLoader config for phase "
                f"'{phase}': persistent_workers=True requires num_workers > 0."
            )

    def check_known_phase(self, phase: str) -> None:
        if phase not in self.datasets:
            raise KeyError(
                f"Unknown phase '{phase}'. "
                f"Available phases are: {sorted(self.datasets.keys())}."
            )
