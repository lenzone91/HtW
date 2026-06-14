from collections.abc import Callable

import torch
from torch import nn

from ...Metrics.ac_video_jepa_objective import AcVideoJepaObjective
from ...Rollouts.latent_rollout import LatentRollout, LatentRolloutOutput
from .base import BaseLightningModule


class AcVideoJepaModule(BaseLightningModule):
    """
    Lightning orchestration for AcVideoJepa.
    """

    required_batch_keys = ("states", "actions")

    def __init__(
        self,
        encoder: nn.Module,
        action_encoder: nn.Module,
        predictor: nn.Module,
        encoder_shape: dict,
        rollout: LatentRollout,
        objective: AcVideoJepaObjective,
        optimizer_builder: Callable,
        scheduler_builder: Callable,
        strict: bool = True,
    ) -> None:
        super().__init__(strict=strict)

        self.encoder = encoder
        self.action_encoder = action_encoder
        self.predictor = predictor
        self.encoder_shape = dict(encoder_shape)
        self.rollout = rollout
        self.objective = objective
        self.optimizer_builder = optimizer_builder
        self.scheduler_builder = scheduler_builder

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
        return self.encoder(observations)

    def encode_actions(self, actions: torch.Tensor | None) -> torch.Tensor | None:
        if actions is None:
            return None

        return self.action_encoder(actions)

    def predict(
        self,
        states: torch.Tensor,
        actions: torch.Tensor | None,
    ) -> torch.Tensor:
        return self.predictor(states, actions)

    def rollout_latents(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
    ) -> LatentRolloutOutput:
        return self.rollout(
            jepa=self,
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
        optimizer = self.optimizer_builder(self.parameters())
        scheduler = self.scheduler_builder(optimizer)

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
