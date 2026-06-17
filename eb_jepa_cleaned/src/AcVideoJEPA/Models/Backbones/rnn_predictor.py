"""
GRU-based single-step latent predictor.

Re-implemented from EB-JEPA (Decision 30). Given a latent state and an action,
predicts the next latent state. The recurrence is rolled out externally (by the
latent rollout), so each call advances exactly one step.

Shape contract:
    state  [B, D, 1, 1, 1]
    action [B, A, 1]
    next   [B, D, 1, 1, 1]
"""

import torch
from torch import nn

from ....AIML.Models.Models.registry import MODEL_REGISTRY

DEFAULT_RNN_PREDICTOR_CONFIG = {
    "hidden_size": 512,
    "action_dim": 2,
    "num_layers": 1,
}


@MODEL_REGISTRY.register_class(
    name="rnn_predictor",
    default_config=DEFAULT_RNN_PREDICTOR_CONFIG,
)
class RNNPredictor(nn.Module):
    """
    Single-step GRU predictor. `is_rnn` / `context_length` flag the rollout that
    this predictor advances one step at a time from a unit context window.

    `final_ln` may be supplied (e.g. to reuse the encoder's final LayerNorm); it
    defaults to identity. It is not a config field — the module wires it when it
    has the encoder (a config carries only `hidden_size`, `action_dim`,
    `num_layers`).
    """

    def __init__(
        self,
        hidden_size: int = 512,
        action_dim: int = 2,
        num_layers: int = 1,
        final_ln: nn.Module | None = None,
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        self.rnn = nn.GRU(
            input_size=action_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
        )
        self.final_ln = nn.Identity() if final_ln is None else final_ln
        self.is_rnn = True
        self.context_length = 0

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        rnn_state = state.flatten(1, 4).unsqueeze(0).contiguous()  # [1, B, D]
        rnn_input = action.squeeze(-1).unsqueeze(0).contiguous()  # [1, B, A]
        next_state, _ = self.rnn(rnn_input, rnn_state)
        next_state = self.final_ln(next_state)
        return next_state[0].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
