from typing import Callable, Optional

import torch
import torch.nn as nn


class NeuralActionSampler(nn.Module):
    """
    Neural sampler for continuous action sequences.

    This module predicts a Gaussian distribution over action sequences
    conditioned on the current observation/context, the planning objective, the
    action bounds, and a variance parameter. It is intentionally independent from
    the planner optimization loop so it can be trained, checkpointed, and reused.
    """

    def __init__(
        self,
        action_dim: int = 2,
        hidden_dim: int = 256,
        min_log_std: float = -5.0,
        max_log_std: float = 2.0,
    ):
        """
        Initialize the neural action sampler.

        Parameters
        ----------
        action_dim : int, default=2
            Dimension of one action vector. For ``two_rooms``, this is 2
            because actions are ``[dx, dy]``.
        hidden_dim : int, default=256
            Width of the MLP used to encode context and predict action
            distribution parameters.
        min_log_std : float, default=-5.0
            Lower clamp value for predicted log standard deviations.
        max_log_std : float, default=2.0
            Upper clamp value for predicted log standard deviations.
        """
        super().__init__()
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.min_log_std = min_log_std
        self.max_log_std = max_log_std

        self.context_net = nn.Sequential(
            nn.LazyLinear(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        self.step_net = nn.Sequential(
            nn.Linear(hidden_dim + 2 + 2 * action_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        self.mu_head = nn.Linear(hidden_dim, action_dim)
        self.log_std_head = nn.Linear(hidden_dim, action_dim)

    def forward(
        self,
        obs_init: torch.Tensor,
        action_low: torch.Tensor,
        action_high: torch.Tensor,
        variance: float,
        objective: Callable,
        plan_length: int,
        num_samples: int,
    ) -> torch.Tensor:
        """
        Sample continuous action sequences from the learned distribution.

        Parameters
        ----------
        obs_init : torch.Tensor
            Current observation/context. Existing planners pass tensors shaped
            ``[B, C, T, H, W]``.
        action_low : torch.Tensor
            Lower action bounds with shape ``[action_dim]``.
        action_high : torch.Tensor
            Upper action bounds with shape ``[action_dim]``.
        variance : float
            External variance multiplier controlling sampling diversity.
        objective : Callable
            Planning objective. If it exposes ``target_enc``, that target
            representation is used as part of the sampler conditioning.
        plan_length : int
            Number of timesteps in each sampled action sequence.
        num_samples : int
            Number of action sequences to sample.

        Returns
        -------
        torch.Tensor
            Sampled action sequences with shape
            ``[num_samples, action_dim, plan_length]`` for a single context, or
            ``[batch_size, num_samples, action_dim, plan_length]`` for batched
            contexts.
        """
        mu, std = self.distribution_parameters(
            obs_init=obs_init,
            action_low=action_low,
            action_high=action_high,
            variance=variance,
            objective=objective,
            plan_length=plan_length,
        )
        batch_size = mu.shape[0]
        eps = torch.randn(
            batch_size,
            num_samples,
            plan_length,
            self.action_dim,
            device=mu.device,
            dtype=mu.dtype,
        )
        actions = mu.unsqueeze(1) + std.unsqueeze(1) * eps
        actions = torch.clamp(
            actions,
            min=action_low.view(1, 1, -1),
            max=action_high.view(1, 1, -1),
        )
        actions = actions.permute(0, 1, 3, 2).contiguous()
        if batch_size == 1:
            return actions.squeeze(0)
        return actions

    def distribution_parameters(
        self,
        obs_init: torch.Tensor,
        action_low: torch.Tensor,
        action_high: torch.Tensor,
        variance: float,
        objective: Optional[Callable],
        plan_length: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Predict Gaussian action-distribution parameters for one planning horizon.

        Parameters
        ----------
        obs_init : torch.Tensor
            Current observation/context used for conditioning.
        action_low : torch.Tensor
            Lower action bounds with shape ``[action_dim]``.
        action_high : torch.Tensor
            Upper action bounds with shape ``[action_dim]``.
        variance : float
            External variance multiplier. The predicted standard deviation is
            multiplied by ``sqrt(variance)``.
        objective : Callable, optional
            Planning objective. ``objective.target_enc`` is used when available.
        plan_length : int
            Number of timesteps for which to predict action parameters.

        Returns
        -------
        tuple of torch.Tensor
            ``(mu, std)`` tensors, each shaped
            ``[batch_size, plan_length, action_dim]``.
        """
        device = next(self.parameters()).device
        obs_init = obs_init.to(device=device)
        action_low, action_high = self._validate_action_inputs(action_low, action_high)
        context = self._build_context(obs_init, objective)
        context = self.context_net(context)
        batch_size = context.shape[0]

        time_features = self._time_features(plan_length, context.device, context.dtype)
        time_features = time_features.view(1, plan_length, -1).expand(
            batch_size, -1, -1
        )
        bounds_features = torch.cat([action_low, action_high], dim=0).view(1, -1)
        bounds_features = bounds_features.view(1, 1, -1).expand(
            batch_size, plan_length, -1
        )
        variance_feature = torch.full(
            (batch_size, plan_length, 1),
            float(variance),
            device=context.device,
            dtype=context.dtype,
        )
        context_features = context.unsqueeze(1).expand(-1, plan_length, -1)
        features = torch.cat(
            [context_features, time_features, variance_feature, bounds_features],
            dim=-1,
        )

        step_features = self.step_net(features)
        mu_unit = torch.tanh(self.mu_head(step_features))
        log_std = self.log_std_head(step_features).clamp(
            self.min_log_std, self.max_log_std
        )

        center = (action_high + action_low) / 2
        radius = (action_high - action_low) / 2
        mu = center.view(1, 1, -1) + radius.view(1, 1, -1) * mu_unit
        std = torch.exp(log_std) * torch.sqrt(
            torch.as_tensor(variance, device=mu.device, dtype=mu.dtype).clamp_min(0.0)
        )
        return mu, std

    def behavior_cloning_loss(
        self,
        obs_init: torch.Tensor,
        target_actions: torch.Tensor,
        action_low: torch.Tensor,
        action_high: torch.Tensor,
        objective: Optional[Callable] = None,
        variance: float = 1.0,
    ) -> torch.Tensor:
        """
        Compute a Gaussian negative log-likelihood imitation loss.

        Parameters
        ----------
        obs_init : torch.Tensor
            Current observation/context used for conditioning.
        target_actions : torch.Tensor
            Demonstration actions with shape ``[T, A]``, ``[A, T]``,
            ``[B, T, A]``, or ``[B, A, T]``.
        action_low : torch.Tensor
            Lower action bounds with shape ``[action_dim]``.
        action_high : torch.Tensor
            Upper action bounds with shape ``[action_dim]``.
        objective : Callable, optional
            Optional objective providing target conditioning.
        variance : float, default=1.0
            External variance multiplier used when computing distribution
            parameters.

        Returns
        -------
        torch.Tensor
            Scalar negative log-likelihood loss.
        """
        target_actions = self._normalize_target_actions(target_actions)
        plan_length = target_actions.shape[-2]
        mu, std = self.distribution_parameters(
            obs_init=obs_init,
            action_low=action_low,
            action_high=action_high,
            variance=variance,
            objective=objective,
            plan_length=plan_length,
        )
        var = std.pow(2).clamp_min(1e-8)
        while target_actions.ndim < mu.ndim:
            target_actions = target_actions.unsqueeze(0)
        nll = 0.5 * (
            (target_actions - mu).pow(2) / var
            + torch.log(var)
            + torch.log(torch.as_tensor(2.0 * torch.pi, device=var.device))
        )
        return nll.mean()

    def _build_context(
        self, obs_init: torch.Tensor, objective: Optional[Callable]
    ) -> torch.Tensor:
        """
        Build a compact conditioning vector from current state and goal context.

        Parameters
        ----------
        obs_init : torch.Tensor
            Current observation/context.
        objective : Callable, optional
            Objective that may expose ``target_enc`` for goal conditioning.

        Returns
        -------
        torch.Tensor
            Context features shaped ``[B, F]``.
        """
        current_features = self._summarize_tensor(obs_init.float())
        target_enc = getattr(objective, "target_enc", None)
        if target_enc is None:
            target_features = torch.zeros_like(current_features)
        else:
            target_features = self._summarize_tensor(
                target_enc.to(device=obs_init.device, dtype=obs_init.dtype).float()
            )
            target_features = self._match_feature_width(target_features, current_features)
            if target_features.shape[0] == 1 and current_features.shape[0] > 1:
                target_features = target_features.expand(current_features.shape[0], -1)
        return torch.cat(
            [
                current_features,
                target_features,
                target_features - current_features,
            ],
            dim=-1,
        )

    def _summarize_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Summarize a state-like tensor with channel-wise mean and standard deviation.

        Parameters
        ----------
        tensor : torch.Tensor
            Tensor shaped ``[B, C, ...]`` or ``[C, ...]``.

        Returns
        -------
        torch.Tensor
            Summary features shaped ``[B, 2 * C]``.
        """
        if tensor.ndim == 0:
            tensor = tensor.view(1, 1)
        elif tensor.ndim == 1:
            tensor = tensor.view(1, -1)
        elif tensor.ndim == 2:
            pass
        else:
            reduce_dims = tuple(range(2, tensor.ndim))
            tensor = torch.cat(
                [
                    tensor.mean(dim=reduce_dims),
                    tensor.std(dim=reduce_dims, unbiased=False),
                ],
                dim=-1,
            )
        return tensor.flatten(start_dim=1)

    def _match_feature_width(
        self, features: torch.Tensor, reference: torch.Tensor
    ) -> torch.Tensor:
        """
        Pad or crop features so they can be combined with a reference tensor.

        Parameters
        ----------
        features : torch.Tensor
            Feature tensor to adapt.
        reference : torch.Tensor
            Feature tensor whose final dimension is the target width.

        Returns
        -------
        torch.Tensor
            Adapted feature tensor with ``reference.shape[-1]`` features.
        """
        target_width = reference.shape[-1]
        if features.shape[-1] == target_width:
            return features
        if features.shape[-1] > target_width:
            return features[..., :target_width]
        padding = features.new_zeros(features.shape[0], target_width - features.shape[-1])
        return torch.cat([features, padding], dim=-1)

    def _time_features(
        self, plan_length: int, device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor:
        """
        Create simple normalized timestep features.

        Parameters
        ----------
        plan_length : int
            Number of timesteps to represent.
        device : torch.device
            Device for the returned tensor.
        dtype : torch.dtype
            Floating dtype for the returned tensor.

        Returns
        -------
        torch.Tensor
            Features shaped ``[plan_length, 1]`` in the range ``[0, 1]``.
        """
        if plan_length == 1:
            return torch.zeros(1, 1, device=device, dtype=dtype)
        return torch.linspace(0, 1, steps=plan_length, device=device, dtype=dtype).view(
            -1, 1
        )

    def _validate_action_inputs(
        self, action_low: torch.Tensor, action_high: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Validate action bounds and move them to the sampler device.

        Parameters
        ----------
        action_low : torch.Tensor
            Lower action bounds.
        action_high : torch.Tensor
            Upper action bounds.

        Returns
        -------
        tuple of torch.Tensor
            Validated ``(action_low, action_high)`` tensors.
        """
        device = next(self.parameters()).device
        action_low = action_low.to(device=device, dtype=torch.float32).flatten()
        action_high = action_high.to(device=device, dtype=torch.float32).flatten()
        if action_low.numel() != self.action_dim or action_high.numel() != self.action_dim:
            raise ValueError(
                "Action bounds must match sampler action_dim "
                f"({action_low.numel()}, {action_high.numel()} != {self.action_dim})"
            )
        return action_low, action_high

    def _normalize_target_actions(self, target_actions: torch.Tensor) -> torch.Tensor:
        """
        Convert target actions to ``[..., T, A]`` format for imitation loss.

        Parameters
        ----------
        target_actions : torch.Tensor
            Target action tensor in one of the accepted action layouts.

        Returns
        -------
        torch.Tensor
            Target action tensor shaped ``[..., T, action_dim]``.
        """
        if target_actions.ndim == 2:
            if target_actions.shape[-1] == self.action_dim:
                return target_actions
            if target_actions.shape[0] == self.action_dim:
                return target_actions.transpose(0, 1)
        if target_actions.ndim == 3:
            if target_actions.shape[-1] == self.action_dim:
                return target_actions
            if target_actions.shape[1] == self.action_dim:
                return target_actions.transpose(1, 2)
        raise ValueError(
            "target_actions must be shaped [T, A], [A, T], [B, T, A], or [B, A, T]"
        )
