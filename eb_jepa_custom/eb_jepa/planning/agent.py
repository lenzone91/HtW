from typing import Callable, Optional

from eb_jepa.logging import get_logger

from .utils import import_from_planning_export, import_from_target
from .utils import objective_name_map, planner_name_map

logger = get_logger(__name__)


class GCAgent:
    def __init__(
        self,
        model,
        action_dim=2,
        plan_cfg=None,
        normalizer: Optional[Callable] = None,
        loc_prober: Optional[Callable] = None,
        img_prober: Optional[Callable] = None,
        env: Optional[Callable] = None,
    ):
        self.plan_cfg = plan_cfg
        self.env = env
        self.model = model
        self.device = next(model.parameters()).device
        self.loc_prober = loc_prober
        self.img_prober = img_prober
        self.normalizer = normalizer

        if plan_cfg is None:
            self.decode_each_iteration = False
            self.num_act_stepped = 1
            self.planner = None
            logger.info("No plan_cfg provided in GCAgent, planner not initialized.")
        else:
            self.decode_each_iteration = plan_cfg.planner.get(
                "decode_each_iteration", False
            )
            self.num_act_stepped = plan_cfg.planner.get("num_act_stepped", 1)
            planner_target = plan_cfg.planner.get("planner_target")
            if planner_target:
                planner_class = import_from_target(planner_target)
            else:
                planner_name = plan_cfg.planner.get("planner_name", "cem")
                planner_class_name = planner_name_map[planner_name]
                planner_class = import_from_planning_export(planner_class_name)
            if planner_class is not None:
                planner_kwargs = dict(plan_cfg.planner)
                planner_kwargs.pop("action_dim", None)
                if (
                    self.env is not None
                    and hasattr(self.env, "action_space")
                    and "action_space" not in planner_kwargs
                    and "action_low" not in planner_kwargs
                    and "action_high" not in planner_kwargs
                ):
                    planner_kwargs["action_space"] = self.env.action_space
                self.planner = planner_class(
                    unroll=self.unroll,
                    action_dim=action_dim,
                    decode_loc_to_pixel=self.decode_loc_to_pixel,
                    **planner_kwargs,
                )
            else:
                logger.info("No planner provided in GCAgent.")
                self.planner = None

        self.goal_state = None
        self.goal_position = None
        self.goal_state_enc = None
        self._prev_losses = None

    def set_goal(self, goal_state, goal_position=None):
        self.goal_position = goal_position
        self.goal_state = goal_state
        self.goal_state_enc = self.model.encode(
            self.normalizer.normalize_state(goal_state.to(self.device))
            .unsqueeze(0)
            .unsqueeze(2)
        )
        objective_name = self.plan_cfg.planner.planning_objective.get(
            "objective_type", "repr_target_dist"
        )
        objective_class_name = objective_name_map[objective_name]
        objective_class = import_from_planning_export(objective_class_name)
        self.objective = objective_class(
            target_enc=self.goal_state_enc, **self.plan_cfg.planner.planning_objective
        )
        self.planner.set_objective(self.objective)

    def unroll(self, obs_init, actions, repeat_batch=True):
        """
        Unroll the model for planning.

        Args:
            obs_init: [B, C, T, H, W]
            actions: [B, A, T]

        Returns:
            predicted_states: [B, D, T, H, W]
        """
        batch_size = actions.shape[0]
        nsteps = actions.shape[2]
        if repeat_batch:
            obs_init = obs_init.repeat(batch_size, 1, 1, 1, 1)
        predicted_states, _ = self.model.unroll(
            obs_init,
            actions,
            nsteps=nsteps,
            unroll_mode="autoregressive",
            ctxt_window_time=self.plan_cfg["ctxt_window_time"] if self.plan_cfg else 1,
            compute_loss=False,
            return_all_steps=False,
        )
        return predicted_states

    def decode_loc_to_pixel(self, predicted_encs, wall_x=None, door_y=None):
        """
        Decode the predicted encodings into frames.

        Args:
            predicted_encs: [B, D, T, H, W]

        Returns:
            np.array of shape [B, T, H, W, C] on cpu for visualization.
        """
        assert self.loc_prober is not None
        B, D, T, H, W = predicted_encs.shape
        out = self.loc_prober.apply_head(predicted_encs).permute(0, 2, 1).cpu()  # B T 2
        out = self.normalizer.unnormalize_location(out)  # B T 2
        frames = self.env.coord_to_pixel(out, wall_x=wall_x, door_y=door_y)  # B T C H W
        frames = frames.permute(0, 1, 3, 4, 2).cpu().numpy()  # B T H W C
        return frames

    def act(self, obs, steps_left=None, t0=False, plan_vis_path=None):
        planning_result = self.planner.plan(
            obs,
            steps_left=steps_left,
            eval_mode=True,
            t0=t0,
            plan_vis_path=plan_vis_path,
        )
        self._prev_losses = planning_result.losses
        self._prev_elite_losses_mean = planning_result.prev_elite_losses_mean
        self._prev_elite_losses_std = planning_result.prev_elite_losses_std
        return planning_result.actions[: self.num_act_stepped]  # T, A
