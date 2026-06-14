import torch

from ..Model.ac_video_jepa.blocks import AcVideoJepaBlocks
from ...Metrics.ac_video_jepa_objective import AcVideoJepaObjective
from ...Rollouts.latent_rollout import LatentRollout, LatentRolloutOutput
from ...Training.Optimization.factory import build_optimizer
from ...Training.Schedulers.factory import build_scheduler
from .base import BaseLightningModule


class AcVideoJepaModule(BaseLightningModule):
    """
    Lightning orchestration for AcVideoJepa.
    """

    required_batch_keys = ("states", "actions")

    def __init__(
        self,
        blocks: AcVideoJepaBlocks,
        rollout: LatentRollout,
        objective: AcVideoJepaObjective,
        optimizer_config: dict,
        scheduler_config: dict | None = None,
        strict: bool = True,
    ) -> None:
        super().__init__(strict=strict)

        self.blocks = blocks
        self.rollout = rollout
        self.objective = objective
        self.optimizer_config = dict(optimizer_config)
        self.scheduler_config = dict(scheduler_config or {"enabled": False})

    def forward(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
    ) -> LatentRolloutOutput:
        return self.rollout_latents(
            observations=observations,
            actions=actions,
        )

    def encode(self, observations: torch.Tensor) -> torch.Tensor:
        return self.blocks.encode(observations)

    def rollout_latents(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
    ) -> LatentRolloutOutput:
        return self.rollout(
            blocks=self.blocks,
            observations=observations,
            actions=actions,
        )

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        loss, log_dict = self.compute_step_loss(batch=batch)
        self.log_step_dict(
            "train",
            log_dict,
            prog_bar=True,
            on_step=True,
            on_epoch=True,
            logger=True,
        )
        return loss

    def validation_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        loss, log_dict = self.compute_step_loss(batch=batch)
        self.log_step_dict(
            "val",
            log_dict,
            prog_bar=False,
            on_step=False,
            on_epoch=True,
            logger=True,
        )
        return loss

    def test_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        loss, log_dict = self.compute_step_loss(batch=batch)
        self.log_step_dict(
            "test",
            log_dict,
            prog_bar=False,
            on_step=False,
            on_epoch=True,
            logger=True,
        )
        return loss

    def compute_step_loss(self, batch: dict) -> tuple[torch.Tensor, dict[str, object]]:
        self.check_batch(batch)

        rollout_output = self.rollout_latents(
            observations=batch["states"],
            actions=batch["actions"],
        )
        total_loss, log_dict = self.objective(
            rollout_output=rollout_output,
            actions=batch["actions"],
        )

        return total_loss, log_dict

    def configure_optimizers(self):
        optimizer = build_optimizer(
            parameters=self.parameters(),
            optimizer_config=self.optimizer_config,
        )
        scheduler = build_scheduler(
            optimizer=optimizer,
            scheduler_config=self.scheduler_config,
        )

        if scheduler is None:
            return optimizer

        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
        }

    def check_batch(self, batch: dict) -> None:
        if not isinstance(batch, dict):
            raise TypeError(
                "AcVideoJepaModule expected batch to be a dictionary, "
                f"got {type(batch).__name__}."
            )

        missing_keys = [
            key
            for key in self.required_batch_keys
            if key not in batch
        ]

        if missing_keys:
            raise KeyError(
                f"AcVideoJepaModule batch is missing required keys: {missing_keys}."
            )

        for key in self.required_batch_keys:
            if not isinstance(batch[key], torch.Tensor):
                raise TypeError(
                    f"AcVideoJepaModule batch key '{key}' must be a torch.Tensor, "
                    f"got {type(batch[key]).__name__}."
                )
