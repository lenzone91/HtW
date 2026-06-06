from .tse_base import TSEBaseLightningModule
from ..Models.waveUnet import WaveUNet

####################################
# WaveUNet lightning module wrapper
####################################




class WaveUNetLightningModule(TSEBaseLightningModule):
    """
    Default TSE Lightning wrapper around WaveUNet.

    Batch convention:
        batch = (mixture, target)

    Model convention:
        preds = WaveUNet(mixture)

    This module is mainly intended as a sanity-check module for the data module
    and the TSE Lightning pipeline.


    Expected __init__ use :

        module = WaveUNetLightningModule(
                    model = waveUnet,
                    optimizer_config=...,
                    scheduler_config=...,
                    train_metrics=...,
                    val_metrics=...,
                    test_metrics=...,
                    loss=...,
                                )

    """

    def __init__(
        self,
        waveunet: WaveUNet,
        **kwargs,
    ):
        
        super().__init__(
            models={"waveunet": waveunet},
            model_name="waveunet",
            **kwargs,
        )

    
    def forward(self, mixture):
        return self.model(mixture)

    def training_step(self, batch, batch_idx: int):
        mixture = batch["mixture"]
        target = batch["target"]

        preds = self(mixture)

        return self.process_tse_step_outputs(
            ml_step="train",
            preds=preds,
            target=target,
            mixture=mixture,
            clue=batch.get("clue", None),
        )

    def validation_step(self, batch, batch_idx: int) -> None:
        mixture = batch["mixture"]
        target = batch["target"]

        preds = self(mixture)

        self.process_tse_step_outputs(
            ml_step="val",
            preds=preds,
            target=target,
            mixture=mixture,
            clue=batch.get("clue", None),
        )

    def test_step(self, batch, batch_idx: int) -> None:
        mixture = batch["mixture"]
        target = batch["target"]

        preds = self(mixture)

        self.process_tse_step_outputs(
            ml_step="test",
            preds=preds,
            target=target,
            mixture=mixture,
            clue=batch.get("clue", None),
        )