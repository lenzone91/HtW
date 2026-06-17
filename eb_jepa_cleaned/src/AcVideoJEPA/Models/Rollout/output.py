"""
Output of a latent rollout.

The structure the JEPA objective metrics consume: the encoded states (targets /
representation to regularize) and the predicted states, plus the metadata a
metric needs to align them (rollout mode, step count, effective context window).

Frozen dataclass so a metric set passes it to each metric as a single argument.
"""

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class LatentRolloutOutput:
    """
    Runtime output of a loss-free latent rollout.

    Fields:
        encoded_states: encoder output over the full clip `[B, D, T, 1, 1]`
            (the prediction targets and the representation the regularizers act on).
        predicted_states: the rolled-out predicted latents.
        actions_encoded: encoded actions used by the rollout (or None).
        nsteps: number of prediction steps rolled out.
        unroll_mode: "parallel" or "autoregressive".
        effective_ctxt_window: context window actually used (1 for an RNN predictor).
        predicted_steps: per-step predictions if `return_all_steps` was set, else None.
    """

    encoded_states: torch.Tensor
    predicted_states: torch.Tensor
    actions_encoded: torch.Tensor | None
    nsteps: int
    unroll_mode: str
    effective_ctxt_window: int
    predicted_steps: list[torch.Tensor] | None = None
