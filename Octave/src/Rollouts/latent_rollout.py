from dataclasses import dataclass

import torch

from .configs import DEFAULT_LATENT_ROLLOUT_CONFIG
from .registry import ROLLOUT_REGISTRY


@dataclass(frozen=True)
class LatentRolloutOutput:
    """
    Runtime output of a loss-free latent rollout.
    """

    encoded_states: torch.Tensor
    predicted_states: torch.Tensor
    actions_encoded: torch.Tensor | None
    nsteps: int
    unroll_mode: str
    effective_ctxt_window: int
    predicted_steps: list[torch.Tensor] | None = None


@ROLLOUT_REGISTRY.register_class(
    name="latent",
    default_config=DEFAULT_LATENT_ROLLOUT_CONFIG,
    type_field="rollout_type",
)
class LatentRollout:
    """
    Loss-free latent rollout over a JEPA runtime object.
    """

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

    def __call__(
        self,
        jepa,
        observations: torch.Tensor,
        actions: torch.Tensor | None,
    ) -> LatentRolloutOutput:
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
        elif self.unroll_mode == "autoregressive":
            effective_ctxt_window = self.get_effective_ctxt_window(jepa)
            predicted_states = self.rollout_autoregressive(
                jepa=jepa,
                encoded_states=encoded_states,
                actions=actions,
                actions_encoded=actions_encoded,
                predicted_steps=predicted_steps,
            )
        else:
            raise ValueError(f"Unknown unroll_mode: {self.unroll_mode}")

        return LatentRolloutOutput(
            encoded_states=encoded_states,
            predicted_states=predicted_states,
            actions_encoded=actions_encoded,
            nsteps=self.nsteps,
            unroll_mode=self.unroll_mode,
            effective_ctxt_window=effective_ctxt_window,
            predicted_steps=predicted_steps,
        )

    def rollout_parallel(
        self,
        jepa,
        encoded_states: torch.Tensor,
        actions_encoded: torch.Tensor | None,
        predicted_steps: list[torch.Tensor] | None,
    ) -> torch.Tensor:
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
        self,
        jepa,
        encoded_states: torch.Tensor,
        actions: torch.Tensor | None,
        actions_encoded: torch.Tensor | None,
        predicted_steps: list[torch.Tensor] | None,
    ) -> torch.Tensor:
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

    def check_config(self) -> None:
        if not isinstance(self.nsteps, int) or self.nsteps < 1:
            raise ValueError("LatentRollout nsteps must be a positive integer.")

        if self.unroll_mode not in self.valid_unroll_modes:
            raise ValueError(
                f"Unknown unroll_mode: {self.unroll_mode}. "
                f"Expected one of {sorted(self.valid_unroll_modes)}."
            )

        if (
            not isinstance(self.ctxt_window_time, int)
            or self.ctxt_window_time < 1
        ):
            raise ValueError(
                "LatentRollout ctxt_window_time must be a positive integer."
            )

    def get_effective_ctxt_window(self, jepa) -> int:
        single_unroll = getattr(jepa.predictor, "is_rnn", False)
        return 1 if single_unroll else self.ctxt_window_time
