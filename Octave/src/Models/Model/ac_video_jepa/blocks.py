from copy import deepcopy

import torch
from torch import nn


class AcVideoJepaBlocks(nn.Module):
    """
    Registered architecture blocks for AcVideoJepa.

    This is a block container, not the semantic JEPA runtime object. Rollout
    behavior and objective computation belong to their own subsystems.
    """

    def __init__(
        self,
        encoder: nn.Module,
        action_encoder: nn.Module,
        predictor: nn.Module,
        encoder_shape: dict[str, int],
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.action_encoder = action_encoder
        self.predictor = predictor
        self.encoder_shape = deepcopy(encoder_shape)

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
