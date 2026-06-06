"""Simple custom planner example loaded through `planner_target`."""

from __future__ import annotations

from typing import Optional

import torch

from eb_jepa.planning import Planner, PlanningResult


class ZeroPlanner(Planner):
    """Planner that always returns zero actions.

    This is intentionally simple: it validates the dynamic planner import path
    and provides a minimal template for future custom planners in `Enzo/`.
    """

    def __init__(
        self,
        unroll,
        plan_length: int = 15,
        action_dim: int = 2,
        **kwargs,
    ):
        super().__init__(unroll=unroll)
        self.plan_length = plan_length
        self.action_dim = action_dim
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @torch.no_grad()
    def plan(
        self,
        obs_init: torch.Tensor,
        steps_left: Optional[int] = None,
        t0: bool = False,
        eval_mode: bool = False,
        plan_vis_path: Optional[str] = None,
    ) -> PlanningResult:
        plan_length = self.plan_length if steps_left is None else min(
            self.plan_length,
            steps_left,
        )
        actions = torch.zeros(plan_length, self.action_dim, device=self.device)
        return PlanningResult(actions=actions, info={"planner": "zero"})

