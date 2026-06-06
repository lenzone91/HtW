from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import torch

from .neural_action_sampler import NeuralActionSampler


@dataclass
class SamplerTrainStepResult:
    """Scalar metrics produced by one sampler optimization step."""

    loss: float
    plan_length: int
    batch_size: int


class _TargetObjective:
    """Minimal objective-like container exposing target_enc to the sampler."""

    def __init__(self, target_enc: torch.Tensor):
        self.target_enc = target_enc


class NeuralActionSamplerTrainer:
    """
    Behavior-cloning trainer for :class:`NeuralActionSampler`.

    The trainer uses batches produced by the existing ``two_rooms`` dataloader:
    ``(states, actions, locations, wall_x, door_y)``. The sampler is trained to
    predict the observed future action sequence from the first observation and a
    goal/target context sampled from the same trajectory.
    """

    def __init__(
        self,
        sampler: NeuralActionSampler,
        optimizer: Optional[torch.optim.Optimizer] = None,
        model: Optional[torch.nn.Module] = None,
        action_low: Optional[torch.Tensor] = None,
        action_high: Optional[torch.Tensor] = None,
        plan_length: Optional[int] = None,
        variance: float = 1.0,
        lr: float = 1e-3,
        grad_clip_norm: Optional[float] = None,
        device: Optional[torch.device | str] = None,
    ):
        """
        Initialize the sampler trainer.

        Parameters
        ----------
        sampler : NeuralActionSampler
            Sampler model to train.
        optimizer : torch.optim.Optimizer, optional
            Optimizer for the sampler. If omitted, an AdamW optimizer is created.
        model : torch.nn.Module, optional
            Optional JEPA model used to encode the goal observation into
            ``target_enc``. Its parameters are not updated by this trainer.
        action_low : torch.Tensor, optional
            Lower action bounds. Defaults to ``[-1, 1]`` according to sampler
            ``action_dim`` when not provided.
        action_high : torch.Tensor, optional
            Upper action bounds. Defaults to ``[1, 1]`` according to sampler
            ``action_dim`` when not provided.
        plan_length : int, optional
            Number of action timesteps used as the behavior-cloning target. When
            omitted, the full available action sequence is used.
        variance : float, default=1.0
            Variance multiplier passed to ``NeuralActionSampler``.
        lr : float, default=1e-3
            Learning rate used when creating the default optimizer.
        grad_clip_norm : float, optional
            Optional gradient norm clipping value.
        device : torch.device or str, optional
            Training device. Defaults to CUDA when available.
        """
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.sampler = sampler.to(self.device)
        self.model = model.to(self.device) if model is not None else None
        if self.model is not None:
            self.model.eval()
        self.optimizer = optimizer or torch.optim.AdamW(
            self.sampler.parameters(), lr=lr
        )
        self.plan_length = plan_length
        self.variance = variance
        self.grad_clip_norm = grad_clip_norm

        if action_low is None:
            action_low = torch.full((self.sampler.action_dim,), -1.0)
        if action_high is None:
            action_high = torch.full((self.sampler.action_dim,), 1.0)
        self.action_low = action_low.to(self.device, dtype=torch.float32)
        self.action_high = action_high.to(self.device, dtype=torch.float32)

    def train_step(self, batch) -> SamplerTrainStepResult:
        """
        Run one behavior-cloning optimization step.

        Parameters
        ----------
        batch : tuple
            Dataloader batch whose first two elements are ``states`` and
            ``actions``. Expected shapes are ``states=[B, C, T, H, W]`` and
            ``actions=[B, A, T]``.

        Returns
        -------
        SamplerTrainStepResult
            Numeric metrics for logging.
        """
        self.sampler.train()
        states, actions = self._extract_states_actions(batch)
        states = states.to(self.device, dtype=torch.float32)
        actions = actions.to(self.device, dtype=torch.float32)
        horizon = self._resolve_plan_length(states, actions)

        obs_init = states[:, :, :1]
        target_actions = actions[:, :, :horizon]
        objective = self._build_target_objective(states, horizon)

        self.optimizer.zero_grad(set_to_none=True)
        loss = self.sampler.behavior_cloning_loss(
            obs_init=obs_init,
            target_actions=target_actions,
            action_low=self.action_low,
            action_high=self.action_high,
            objective=objective,
            variance=self.variance,
        )
        loss.backward()
        if self.grad_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(
                self.sampler.parameters(), self.grad_clip_norm
            )
        self.optimizer.step()

        return SamplerTrainStepResult(
            loss=float(loss.detach().cpu()),
            plan_length=horizon,
            batch_size=states.shape[0],
        )

    def fit(
        self,
        loader: Iterable,
        epochs: int = 1,
        log_every: Optional[int] = None,
    ) -> list[SamplerTrainStepResult]:
        """
        Train the sampler for multiple epochs over a dataloader.

        Parameters
        ----------
        loader : Iterable
            Iterable returning training batches.
        epochs : int, default=1
            Number of full passes over ``loader``.
        log_every : int, optional
            If provided, prints a compact progress line every ``log_every``
            optimization steps.

        Returns
        -------
        list of SamplerTrainStepResult
            Per-step training metrics.
        """
        history = []
        global_step = 0
        for epoch in range(epochs):
            for batch in loader:
                result = self.train_step(batch)
                history.append(result)
                if log_every is not None and global_step % log_every == 0:
                    print(
                        "sampler_train "
                        f"epoch={epoch} step={global_step} loss={result.loss:.6f}"
                    )
                global_step += 1
        return history

    def save_checkpoint(self, path: str | Path):
        """
        Save sampler and optimizer state.

        Parameters
        ----------
        path : str or pathlib.Path
            Destination checkpoint path.
        """
        torch.save(
            {
                "sampler": self.sampler.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "plan_length": self.plan_length,
                "variance": self.variance,
                "action_low": self.action_low,
                "action_high": self.action_high,
            },
            path,
        )

    def load_checkpoint(self, path: str | Path):
        """
        Load sampler and optimizer state.

        Parameters
        ----------
        path : str or pathlib.Path
            Source checkpoint path.
        """
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.sampler.load_state_dict(checkpoint["sampler"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.plan_length = checkpoint.get("plan_length", self.plan_length)
        self.variance = checkpoint.get("variance", self.variance)
        self.action_low = checkpoint.get("action_low", self.action_low).to(self.device)
        self.action_high = checkpoint.get("action_high", self.action_high).to(
            self.device
        )

    def _extract_states_actions(self, batch) -> tuple[torch.Tensor, torch.Tensor]:
        if not isinstance(batch, (tuple, list)) or len(batch) < 2:
            raise ValueError("batch must contain at least states and actions")
        return batch[0], batch[1]

    def _resolve_plan_length(
        self, states: torch.Tensor, actions: torch.Tensor
    ) -> int:
        available = min(states.shape[2], actions.shape[2])
        if self.plan_length is None:
            return available
        return min(self.plan_length, available)

    @torch.no_grad()
    def _build_target_objective(
        self, states: torch.Tensor, horizon: int
    ) -> _TargetObjective:
        target_idx = max(0, horizon - 1)
        target_obs = states[:, :, target_idx : target_idx + 1]
        if self.model is None:
            target_enc = target_obs
        else:
            target_enc = self.model.encode(target_obs)
        return _TargetObjective(target_enc=target_enc.detach())
