"""
Inverse dynamics model.

Re-implemented from EB-JEPA (Decision 30). Predicts the action that caused a
transition from one latent state to the next, from the concatenation of the two
states. Used as the auxiliary inverse-dynamics regularizer.

Plain building block (not a registered model): the inverse-dynamics regularizer
metric constructs it via its own field resolver (it depends on the encoder
feature dimension, resolved at module-build time).
"""

import torch
from torch import nn


def init_module_weights(module: nn.Module, std: float = 0.02) -> None:
    """Truncated-normal init for conv/linear weights; zero bias."""
    conv_linear = (
        nn.Conv2d,
        nn.Conv3d,
        nn.ConvTranspose2d,
        nn.ConvTranspose3d,
        nn.Linear,
    )
    if isinstance(module, conv_linear):
        nn.init.trunc_normal_(module.weight, std=std)
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)


class InverseDynamicsModel(nn.Module):
    """MLP over concatenated states `[B, 2*D] -> [B, A]`."""

    def __init__(self, state_dim: int, hidden_dim: int, action_dim: int) -> None:
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(state_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )
        self.apply(init_module_weights)

    def forward(
        self,
        state_t: torch.Tensor,
        state_t_plus_1: torch.Tensor,
    ) -> torch.Tensor:
        combined_states = torch.cat([state_t, state_t_plus_1], dim=1)
        return self.model(combined_states)
