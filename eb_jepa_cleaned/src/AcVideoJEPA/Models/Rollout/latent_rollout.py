"""
Loss-free latent rollout.

Re-implemented from EB-JEPA's `ac_video_jepa` (Decision 30). Given a JEPA runtime
object exposing `encode`, `encode_actions`, `predict`, and a `predictor`, it
encodes a clip and rolls the predictor forward in latent space, returning a
`LatentRolloutOutput`. It computes no loss — the metrics consume its output.

Two modes:
- "parallel": predict all steps from the encoded context each step;
- "autoregressive": feed predictions back in, one step at a time.
"""

import torch

from .output import LatentRolloutOutput
from .registry import ROLLOUT_REGISTRY

DEFAULT_LATENT_ROLLOUT_CONFIG = {
    "rollout_type": "latent",
    "nsteps": 1,
    "unroll_mode": "autoregressive",
    "ctxt_window_time": 1,
    "return_all_steps": False,
}


@ROLLOUT_REGISTRY.register_class(
    name="latent",
    default_config=DEFAULT_LATENT_ROLLOUT_CONFIG,
    type_field="rollout_type",
)
class LatentRollout:
    """Loss-free latent rollout over a JEPA runtime object."""

    valid_unroll_modes = {"parallel", "autoregressive"}

    def __init__(
        self,
        nsteps: int,
        unroll_mode: str,
        ctxt_window_time: int = 1,
        return_all_steps: bool = False,
    ) -> None:
        self.nsteps = nsteps
        self.unroll_mode = unroll_mode
        self.ctxt_window_time = ctxt_window_time
        self.return_all_steps = return_all_steps
        self.check_config()

    def __call__(self, jepa, observations, actions) -> LatentRolloutOutput:
        encoded_states = jepa.encode(observations)
        actions_encoded = jepa.encode_actions(actions)
        predicted_steps = [] if self.return_all_steps else None

        if self.unroll_mode == "parallel":
            effective_ctxt_window = getattr(jepa.predictor, "context_length", 0)
            predicted_states = self.rollout_parallel(
                jepa=jepa,
                encoded_states=encoded_states,
                actions_encoded=actions_encoded,
                predicted_steps=predicted_steps,
            )
        else:  # "autoregressive" (validated in check_config)
            effective_ctxt_window = self.get_effective_ctxt_window(jepa)
            predicted_states = self.rollout_autoregressive(
                jepa=jepa,
                encoded_states=encoded_states,
                actions=actions,
                actions_encoded=actions_encoded,
                predicted_steps=predicted_steps,
            )

        return LatentRolloutOutput(
            encoded_states=encoded_states,
            predicted_states=predicted_states,
            actions_encoded=actions_encoded,
            nsteps=self.nsteps,
            unroll_mode=self.unroll_mode,
            effective_ctxt_window=effective_ctxt_window,
            predicted_steps=predicted_steps,
        )

    def rollout_parallel(self, jepa, encoded_states, actions_encoded, predicted_steps):
        context_length = getattr(jepa.predictor, "context_length", 0)
        predicted_states = encoded_states

        for _ in range(self.nsteps):
            predicted_states = jepa.predict(predicted_states, actions_encoded)[
                :, :, :-1
            ]
            if predicted_steps is not None:
                predicted_steps.append(predicted_states)

            predicted_states = torch.cat(
                (encoded_states[:, :, :context_length], predicted_states),
                dim=2,
            )

        return predicted_states

    def rollout_autoregressive(
        self, jepa, encoded_states, actions, actions_encoded, predicted_steps
    ):
        if actions is not None and self.nsteps > actions.size(2):
            raise ValueError(
                f"nsteps ({self.nsteps}) larger than action sequence length "
                f"({actions.size(2)})"
            )

        effective_ctxt_window = self.get_effective_ctxt_window(jepa)
        predicted_states = encoded_states[:, :, :effective_ctxt_window]

        for step_index in range(self.nsteps):
            context_states = predicted_states[:, :, -effective_ctxt_window:]
            if actions_encoded is not None:
                context_actions = actions_encoded[
                    :,
                    :,
                    max(0, step_index + 1 - effective_ctxt_window) : step_index + 1,
                ]
            else:
                context_actions = None

            pred_step = jepa.predict(context_states, context_actions)[:, :, -1:]
            predicted_states = torch.cat([predicted_states, pred_step], dim=2)

            if predicted_steps is not None:
                predicted_steps.append(predicted_states.clone())

        return predicted_states

    #############################################
    # Validation
    #############################################

    def check_config(self) -> None:
        if not isinstance(self.nsteps, int) or self.nsteps < 1:
            raise ValueError("LatentRollout nsteps must be a positive integer.")

        if self.unroll_mode not in self.valid_unroll_modes:
            raise ValueError(
                f"Unknown unroll_mode: {self.unroll_mode}. "
                f"Expected one of {sorted(self.valid_unroll_modes)}."
            )

        if not isinstance(self.ctxt_window_time, int) or self.ctxt_window_time < 1:
            raise ValueError(
                "LatentRollout ctxt_window_time must be a positive integer."
            )

    def get_effective_ctxt_window(self, jepa) -> int:
        single_unroll = getattr(jepa.predictor, "is_rnn", False)
        return 1 if single_unroll else self.ctxt_window_time
