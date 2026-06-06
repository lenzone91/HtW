from typing import Callable, List, Optional

import numpy as np
import torch
from einops import rearrange

from eb_jepa.vis_utils import save_decoded_frames

from .base import Planner, PlanningResult


class MPPIPlanner(Planner):
    def __init__(
        self,
        unroll: Callable,
        n_iters: int = 15,
        num_samples: int = 500,
        plan_length: int = 15,
        action_dim: int = 2,
        max_std: float = 2,
        num_elites: int = 64,
        temperature: float = 0.005,
        max_norms: Optional[List[float]] = None,
        max_norm_dims: Optional[List[List[int]]] = None,
        decode_each_iteration: bool = False,
        decode_loc_to_pixel: Optional[Callable] = None,
        **kwargs,
    ):
        super().__init__(unroll)
        self.n_iters = n_iters
        self.num_samples = num_samples
        self.plan_length = plan_length
        self.action_dim = action_dim
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.max_std = max_std
        self.num_elites = num_elites
        self.temperature = temperature
        self.max_norms = max_norms
        self.max_norm_dims = max_norm_dims
        self.decode_each_iteration = decode_each_iteration
        self.decode_loc_to_pixel = decode_loc_to_pixel
        self._prev_mean = None

    @torch.no_grad()
    def plan(
        self, obs_init, t0=False, eval_mode=False, steps_left=None, plan_vis_path=None
    ):
        """
        Args:
                obs_init (torch.Tensor): Latent state from which to plan.
                t0 (bool): Whether this is the first observation in the episode.
                eval_mode (bool): Whether to use the mean of the action distribution.
                task (Torch.Tensor): Task index (only used for multi-task experiments).

        Returns:
                torch.Tensor: Action to take in the environment.
        """
        if steps_left is None:
            plan_length = self.plan_length
        else:
            plan_length = min(self.plan_length, steps_left)

        mean = torch.zeros(plan_length, self.action_dim, device=self.device)
        std = self.max_std * torch.ones(
            plan_length, self.action_dim, device=self.device
        )
        actions = torch.empty(
            plan_length,
            self.num_samples,
            self.action_dim,
            device=self.device,
        )

        losses = []
        elite_means = []
        elite_stds = []
        if self.decode_each_iteration:
            pred_frames_over_iterations = []

        for _ in range(self.n_iters):
            actions[:, :] = mean.unsqueeze(1) + std.unsqueeze(1) * torch.randn(
                plan_length,
                self.num_samples,
                self.action_dim,
                device=std.device,
            )  # T B A
            cost = self.cost_function(
                rearrange(actions, "t b a -> b a t"), obs_init
            ).unsqueeze(1)
            losses.append(cost.min().item())

            elite_idxs = torch.topk(-cost.squeeze(1), self.num_elites, dim=0).indices
            elite_loss, elite_actions = cost[elite_idxs], actions[:, elite_idxs]

            elite_means.append(elite_loss.mean().item())
            elite_stds.append(elite_loss.std().item())

            min_cost = cost.min(0)[0]
            score = torch.exp(
                self.temperature * (min_cost - elite_loss[:, 0])
            )  # increasing with elite_value
            score /= score.sum(0)
            mean = torch.sum(
                score.unsqueeze(0).unsqueeze(2) * elite_actions, dim=1
            ) / (  # T B A
                score.sum(0) + 1e-9
            )
            std = torch.sqrt(
                torch.sum(
                    score.unsqueeze(0).unsqueeze(2)
                    * (elite_actions - mean.unsqueeze(1)) ** 2,
                    dim=1,  # T B A
                )
                / (score.sum(0) + 1e-9)
            )
            if self.decode_each_iteration:
                predicted_best_encs = self.unroll(
                    obs_init, rearrange(mean, "t a -> 1 a t")
                )
                pred_frames = self.decode_loc_to_pixel(
                    predicted_best_encs,
                )
                pred_frames_over_iterations.append(pred_frames.squeeze(0))

        if self.decode_each_iteration:
            save_decoded_frames(pred_frames_over_iterations, losses, plan_vis_path)

        score = score.cpu().numpy()
        actions = elite_actions[
            :, np.random.choice(np.arange(score.shape[0]), p=score)
        ]  # T, A
        self._prev_mean = mean
        if not eval_mode:
            actions += std * torch.randn(
                self.action_dim, device=std.device, generator=self.local_generator
            )

        return PlanningResult(
            actions=actions,
            losses=torch.tensor(losses).detach().unsqueeze(-1),
            prev_elite_losses_mean=torch.tensor(elite_means).unsqueeze(-1),
            prev_elite_losses_std=torch.tensor(elite_stds).unsqueeze(-1),
        )
