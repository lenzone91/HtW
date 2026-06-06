from typing import Callable, Optional

import torch

from .base import Planner, PlanningResult
from .utils import import_from_target


class NeuralStochasticPlanner(Planner):
    """
    Planner skeleton for a future learned stochastic action distribution.

    The future sampler/model is expected to propose action sequences conditioned on
    the current state, the continuous action space, a variance parameter, and the
    planning objective.
    """

    def __init__(
        self,
        unroll: Callable,
        plan_length: int = 15,
        action_dim: int = 2,
        num_samples: int = 256,
        variance: float = 1.0,
        action_space: Optional[dict] = None,
        action_low: Optional[list[float]] = None,
        action_high: Optional[list[float]] = None,
        max_norms: Optional[list[float] | str] = "auto",
        max_norm_dims: Optional[list[list[int]]] = None,
        sampler_target: Optional[str] = None,
        **kwargs,
    ):
        """
        Initialize the stochastic planner and infer action constraints.

        Parameters
        ----------
        unroll : Callable
            Function used to roll out latent states from an initial observation
            and candidate action sequences. It must accept actions shaped
            ``[num_samples, action_dim, plan_length]``.
        plan_length : int, default=15
            Maximum number of future timesteps planned at each call to
            :meth:`plan`.
        action_dim : int, default=2
            Fallback action dimension used only when no action bounds can be
            inferred from ``action_space``, ``action_low``, or ``action_high``.
        num_samples : int, default=256
            Number of candidate action sequences to sample and score.
        variance : float, default=1.0
            Variance parameter passed to the learned sampler or used by the
            fallback random sampler.
        action_space : dict or gym.Space, optional
            Continuous action space. A Gym-like space should expose ``low`` and
            ``high`` attributes. A dict should contain ``"low"`` and ``"high"``.
        action_low : list of float, optional
            Lower action bounds used when ``action_space`` is not provided.
        action_high : list of float, optional
            Upper action bounds used when ``action_space`` is not provided.
        max_norms : list of float, "auto", or None, default="auto"
            Optional max-norm constraints. ``"auto"`` infers a single norm limit
            from symmetric box bounds such as ``[-2.45, 2.45]``.
        max_norm_dims : list of list of int, optional
            Action dimensions associated with each max-norm constraint. The
            CEM-style shorthand ``[0, 1]`` is accepted when there is one norm.
        sampler_target : str, optional
            Dotted Python target for a future learned sampler class. When
            omitted, the planner uses the random fallback sampler.
        **kwargs
            Extra planner configuration values accepted for compatibility with
            shared YAML configs.
        """
        super().__init__(unroll)
        self.plan_length = plan_length
        self.num_samples = num_samples
        self.variance = variance
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.action_sampler = (
            import_from_target(sampler_target)() if sampler_target is not None else None
        )

        action_low, action_high = self._extract_action_bounds(
            action_space=action_space,
            action_low=action_low,
            action_high=action_high,
        )
        if action_low is None:
            action_low = [-1.0] * action_dim
        if action_high is None:
            action_high = [1.0] * action_dim

        self.action_low = torch.as_tensor(
            action_low, dtype=torch.float32, device=self.device
        )
        self.action_high = torch.as_tensor(
            action_high, dtype=torch.float32, device=self.device
        )
        self.action_dim = self.action_low.numel()
        self._validate_action_bounds()
        self.max_norms = self._resolve_max_norms(max_norms)
        self.max_norm_dims = max_norm_dims

    @torch.no_grad()
    def plan(
        self, obs_init, steps_left=None, eval_mode=True, t0=False, plan_vis_path=None
    ):
        """
        Sample candidate action sequences, score them with the planning objective,
        and return the best sequence in the standard [T, A] planner format.

        Parameters
        ----------
        obs_init : torch.Tensor
            Initial observation or latent context passed to ``self.unroll``.
            Existing planners use shape ``[B, C, T, H, W]``.
        steps_left : int, optional
            Number of environment steps remaining in the episode. When provided,
            the planning horizon is clipped to ``min(plan_length, steps_left)``.
        eval_mode : bool, default=True
            Kept for compatibility with the base planner interface. The current
            implementation always returns the best-scoring sequence.
        t0 : bool, default=False
            Whether the planner is called at the first timestep of an episode.
            Kept for interface compatibility.
        plan_vis_path : str or pathlib.Path, optional
            Optional path for planner visualizations. Currently unused.

        Returns
        -------
        PlanningResult
            Result containing the selected action sequence with shape ``[T, A]``,
            all sampled sequence costs, and lightweight diagnostic info.
        """
        if steps_left is None:
            plan_length = self.plan_length
        else:
            plan_length = min(self.plan_length, steps_left)

        actions = self.sample_action_sequences(
            obs_init=obs_init,
            action_low=self.action_low,
            action_high=self.action_high,
            variance=self.variance,
            objective=self.objective,
            plan_length=plan_length,
            num_samples=self.num_samples,
        )
        costs = self.cost_function(actions, obs_init)
        best_idx = torch.argmin(costs)
        best_actions = actions[best_idx].transpose(0, 1)  # [A, T] -> [T, A]

        return PlanningResult(
            actions=best_actions,
            losses=costs.detach(),
            info={
                "best_cost": costs[best_idx].detach(),
                "variance": self.variance,
            },
        )

    def sample_action_sequences(
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
        Return action sequences with shape [num_samples, action_dim, plan_length].

        This is the integration point for the future learned sampler. Until a
        sampler is provided, the fallback draws candidate actions and adds
        Gaussian noise controlled by `variance`.

        Parameters
        ----------
        obs_init : torch.Tensor
            Initial observation or latent context used to condition the sampler.
        action_low : torch.Tensor
            Lower action bounds with shape ``[action_dim]``.
        action_high : torch.Tensor
            Upper action bounds with shape ``[action_dim]``.
        variance : float
            Variance parameter controlling sampler diversity.
        objective : Callable
            Planning objective used by the future learned sampler. The fallback
            sampler does not use it directly.
        plan_length : int
            Number of timesteps in each sampled action sequence.
        num_samples : int
            Number of action sequences to sample.

        Returns
        -------
        torch.Tensor
            Candidate action sequences with shape
            ``[num_samples, action_dim, plan_length]``.
        """
        if self.action_sampler is not None:
            return self.action_sampler(
                obs_init=obs_init,
                action_low=action_low,
                action_high=action_high,
                variance=variance,
                objective=objective,
                plan_length=plan_length,
                num_samples=num_samples,
            )
        return self._sample_random_action_sequences(
            action_low=action_low,
            action_high=action_high,
            variance=variance,
            plan_length=plan_length,
            num_samples=num_samples,
        )

    def _validate_action_bounds(self):
        """
        Check that low/high action bounds match the inferred action dimension.

        Raises
        ------
        ValueError
            If bound shapes do not match ``action_dim`` or if any lower bound is
            greater than or equal to its corresponding upper bound.
        """
        if self.action_low.shape != (self.action_dim,):
            raise ValueError(
                f"action_low must have shape [{self.action_dim}], "
                f"got {tuple(self.action_low.shape)}"
            )
        if self.action_high.shape != (self.action_dim,):
            raise ValueError(
                f"action_high must have shape [{self.action_dim}], "
                f"got {tuple(self.action_high.shape)}"
            )
        if torch.any(self.action_low >= self.action_high):
            raise ValueError("Each action_low value must be smaller than action_high")

    def _sample_random_action_sequences(
        self,
        action_low: torch.Tensor,
        action_high: torch.Tensor,
        variance: float,
        plan_length: int,
        num_samples: int,
    ) -> torch.Tensor:
        """
        Fallback sampler used before the learned model exists.

        It samples continuous actions uniformly inside the environment bounds,
        perturbs them with Gaussian noise controlled by `variance`, then enforces
        both box bounds and optional norm constraints.

        Parameters
        ----------
        action_low : torch.Tensor
            Lower action bounds with shape ``[action_dim]``.
        action_high : torch.Tensor
            Upper action bounds with shape ``[action_dim]``.
        variance : float
            Variance of the Gaussian perturbation applied after uniform sampling.
        plan_length : int
            Number of timesteps in each sampled action sequence.
        num_samples : int
            Number of action sequences to sample.

        Returns
        -------
        torch.Tensor
            Sampled action sequences with shape
            ``[num_samples, action_dim, plan_length]``.
        """
        shape = (num_samples, plan_length, self.action_dim)
        actions = (
            action_low
            + torch.rand(shape, device=self.device) * (action_high - action_low)
        )
        std = torch.sqrt(torch.as_tensor(variance, device=self.device).clamp_min(0.0))
        actions = actions + torch.randn_like(actions) * std
        actions = torch.clamp(
            actions,
            min=action_low.view(1, 1, -1),
            max=action_high.view(1, 1, -1),
        )
        actions = self._project_action_norms(actions)
        actions = torch.clamp(
            actions,
            min=action_low.view(1, 1, -1),
            max=action_high.view(1, 1, -1),
        )
        return actions.permute(0, 2, 1).contiguous()  # [B, A, T]

    def _extract_action_bounds(self, action_space, action_low, action_high):
        """
        Resolve action bounds from explicit config or from a Gym-like action_space.

        Explicit `action_low` / `action_high` values remain a fallback, but when
        `GCAgent` provides an environment action_space this lets the planner adapt
        to different environments without hard-coded limits in YAML.

        Parameters
        ----------
        action_space : dict or gym.Space or None
            Optional source of action bounds. Dicts are expected to contain
            ``"low"`` and ``"high"`` keys; Gym-like spaces expose ``low`` and
            ``high`` attributes.
        action_low : list of float or array-like or None
            Explicit lower bounds used if ``action_space`` does not provide them.
        action_high : list of float or array-like or None
            Explicit upper bounds used if ``action_space`` does not provide them.

        Returns
        -------
        tuple
            ``(action_low, action_high)`` bounds, possibly still ``None`` if no
            source provided them.
        """
        if action_space is None:
            return action_low, action_high
        if isinstance(action_space, dict):
            return (
                action_space.get("low", action_low),
                action_space.get("high", action_high),
            )
        return (
            getattr(action_space, "low", action_low),
            getattr(action_space, "high", action_high),
        )

    def _project_action_norms(self, actions: torch.Tensor) -> torch.Tensor:
        """
        Project actions onto max-norm balls, matching the clipping logic used by CEM.

        `actions` is shaped [num_samples, plan_length, action_dim]. Each configured
        dimension group is independently projected if its norm exceeds max_norm.

        Parameters
        ----------
        actions : torch.Tensor
            Candidate actions with shape
            ``[num_samples, plan_length, action_dim]``.

        Returns
        -------
        torch.Tensor
            Projected actions with the same shape as ``actions``.
        """
        if self.max_norms is None:
            return actions

        projected = actions
        norm_dims = self._normalize_max_norm_dims()
        for max_norm, dims in zip(self.max_norms, norm_dims):
            dims_tensor = torch.as_tensor(dims, dtype=torch.long, device=self.device)
            selected = projected.index_select(dim=-1, index=dims_tensor)
            norms = selected.norm(dim=-1, keepdim=True)
            coeff = torch.where(
                norms > max_norm,
                torch.as_tensor(max_norm, device=self.device) / (norms + 1e-6),
                torch.ones_like(norms),
            )
            projected = projected.clone()
            projected[..., dims] = selected * coeff
        return projected

    def _resolve_max_norms(self, max_norms):
        """
        Convert the `max_norms` setting into concrete norm limits.

        The default value "auto" infers a single norm limit from symmetric box
        bounds, e.g. Box(-2.45, 2.45, shape=(2,)) becomes [2.45].
        Asymmetric action spaces keep norm projection disabled unless configured.

        Parameters
        ----------
        max_norms : list of float, "auto", or None
            Norm constraint configuration from the planner YAML or constructor.

        Returns
        -------
        list of float or None
            Concrete norm limits, or ``None`` when norm projection should be
            disabled.
        """
        if max_norms != "auto":
            return max_norms
        if torch.allclose(self.action_low, -self.action_high):
            return [float(self.action_high.abs().min().item())]
        return None

    def _normalize_max_norm_dims(self) -> list[list[int]]:
        """
        Normalize max_norm_dims to one list of dimensions per norm constraint.

        This accepts the existing CEM-style YAML shorthand:
        max_norms: [2.45], max_norm_dims: [0, 1].

        Returns
        -------
        list of list of int
            One dimension group per entry in ``self.max_norms``.
        """
        if self.max_norm_dims is None:
            return [list(range(self.action_dim)) for _ in self.max_norms]
        if len(self.max_norms) == 1 and all(
            isinstance(dim, int) for dim in self.max_norm_dims
        ):
            return [self.max_norm_dims]
        return self.max_norm_dims
