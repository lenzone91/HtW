import random as py_random

import numpy as np
import torch

from environments.two_rooms.wall_dataset import WallDataset, WallSample
from environments.multiroom.config import MultiRoomConfig, sample_multiroom_walls
from environments.multiroom.physics import check_multi_wall_step
from environments.multiroom.renderer import render_walls_multi


class MultiRoomDataset(WallDataset):
    """Dataset for the MultiRoom environment.

    Generates on-the-fly trajectories in an arena with N rooms separated
    by N-1 vertical walls, each with a door.

    Output format is identical to WallDataset / TwoRooms — the EB-JEPA
    training loop sees no difference.

    Args:
        config: MultiRoomConfig instance

    Output per sample (WallSample):
        states:    (2, sample_length, img_size, img_size) — [dot channel, walls channel]
        actions:   (2, sample_length)
        locations: (2, sample_length) — normalized (x, y) positions
        wall_x:    (num_walls,) — wall x positions (extended metadata)
        door_y:    (num_walls,) — door y positions (extended metadata)
    """

    def __init__(self, config: MultiRoomConfig):
        super().__init__(config)

    # ------------------------------------------------------------------
    # Wall sampling — N-1 walls instead of 1
    # ------------------------------------------------------------------

    def sample_walls(self):
        return sample_multiroom_walls(self.config, device=self.device)

    # ------------------------------------------------------------------
    # Start position — avoid all walls
    # ------------------------------------------------------------------

    def generate_state(self, wall_locs=None, door_locs=None, size=None):
        """Generate a start position always in the leftmost room."""
        if size is None:
            size = 1

        border = float(self.config.border_wall_loc)
        img_size = float(self.config.img_size)
        half_w = self.config.wall_width // 2

        # x is bounded by the left border and the left face of the first wall
        x_min = border
        x_max = float(wall_locs[0].item()) - half_w - 1.0
        if x_max <= x_min:
            x_max = x_min + 1.0

        y_min = border
        y_max = img_size - border

        x = torch.rand(size=(size, 1), device=self.device) * (x_max - x_min) + x_min
        y = torch.rand(size=(size, 1), device=self.device) * (y_max - y_min) + y_min
        location = torch.cat([x, y], dim=1)

        return location

    def generate_state_and_actions(self, wall_locs=None, door_locs=None, size=None, n_steps=17):
        """Override to handle multi-wall cross trajectories.

        Bypasses WallDataset's single-wall cross logic entirely.
        With probability cross_wall_rate, generates a trajectory that goes from
        the leftmost room to the rightmost room through all doors in sequence.
        """
        location = self.generate_state(wall_locs=wall_locs, door_locs=door_locs)
        actions, bias_angle = self.generate_actions(n_steps=n_steps)

        cross_wall_rate = getattr(self.config, "cross_wall_rate", 0.0)
        if cross_wall_rate > 0 and np.random.rand() < cross_wall_rate:
            cw_loc, cw_actions = self.generate_multi_cross_wall_state_and_actions(
                wall_xs=wall_locs, door_ys=door_locs, n_steps=n_steps
            )
            if cw_loc is not None:
                location = cw_loc
                actions = cw_actions

        return location, actions, bias_angle

    def generate_multi_cross_wall_state_and_actions(self, wall_xs, door_ys, n_steps):
        """Generate a trajectory that crosses all walls from left to right.

        Path: room_0_start → door_0 → door_1 → ... → room_N_goal
        Uses generate_actions_to_goal (expert straight-line actions) per segment,
        then pads the remaining steps with a random Von Mises walk.

        Args:
            wall_xs:  (num_walls,) x-coordinates of walls
            door_ys:  (num_walls,) y-coordinates of door centers
            n_steps:  total trajectory length (returns n_steps-1 actions)

        Returns:
            (start_loc (1,2), actions (1, n_steps-1, 2)) or (None, None) on failure
        """
        config = self.config
        border = float(config.border_wall_loc)
        img_size = float(config.img_size)
        half_w = config.wall_width // 2

        wall_xs_list = wall_xs.tolist()
        door_ys_list = door_ys.tolist()

        # --- Start position: in the leftmost room ---
        max_start_x = wall_xs_list[0] - half_w - 3.0
        min_start_x = border + 1.0
        if max_start_x <= min_start_x:
            return None, None
        start_x = py_random.uniform(min_start_x, max_start_x)
        start_y = py_random.uniform(border + 1.0, img_size - border - 1.0)
        start_pos = torch.tensor([start_x, start_y], device=self.device)

        # --- Goal position: in the rightmost room ---
        min_goal_x = wall_xs_list[-1] + half_w + 3.0
        max_goal_x = img_size - border - 1.0
        if min_goal_x >= max_goal_x:
            return None, None
        goal_x = py_random.uniform(min_goal_x, max_goal_x)
        goal_y = py_random.uniform(border + 1.0, img_size - border - 1.0)
        goal_pos = torch.tensor([goal_x, goal_y], device=self.device)

        # --- Waypoints: start → (approach_wall, enter_next_room) × N → goal ---
        # Two waypoints per wall: just before the left face, just after the right face.
        # This guarantees the agent physically crosses into each room (x > wall_x + half_w).
        waypoints = [start_pos]
        for wx, dy in zip(wall_xs_list, door_ys_list):
            # Approach: just outside the left face of the wall, aligned with the door
            before_wall = torch.tensor(
                [wx - half_w - 1.0, dy], dtype=torch.float32, device=self.device
            )
            # Entry: just inside the right face, confirming room entry
            after_wall = torch.tensor(
                [wx + half_w + 1.0, dy], dtype=torch.float32, device=self.device
            )
            waypoints.append(before_wall)
            waypoints.append(after_wall)
        waypoints.append(goal_pos)

        # --- Expert straight-line actions between each pair of waypoints ---
        all_actions = []
        curr_pos = waypoints[0]
        for wp in waypoints[1:]:
            seg = self.generate_actions_to_goal(curr_pos, wp)
            if len(seg) > 0:
                all_actions.append(seg)
            curr_pos = wp

        if not all_actions:
            return None, None

        actions_tensor = torch.cat(all_actions, dim=0)  # (path_steps, 2)

        # --- Pad with a random walk from goal if path is shorter than n_steps-1 ---
        if actions_tensor.shape[0] < n_steps - 1:
            extra_needed = (n_steps - 1) - actions_tensor.shape[0]
            # Generate extra random Von Mises actions (adds natural wandering at end)
            extra_actions, _ = self.generate_actions(n_steps=extra_needed + 2)
            actions_tensor = torch.cat([actions_tensor, extra_actions[0, :extra_needed]], dim=0)

        if actions_tensor.shape[0] < n_steps - 1:
            return None, None

        # Take exactly n_steps-1 actions (crossing is at the start of the trajectory)
        sampled_actions = actions_tensor[: n_steps - 1].unsqueeze(0)  # (1, n_steps-1, 2)

        return start_pos.unsqueeze(0), sampled_actions

    # ------------------------------------------------------------------
    # Wall rendering — all walls on one canvas
    # ------------------------------------------------------------------

    def render_walls(self, wall_locs, hole_locs):
        return render_walls_multi(
            wall_xs=wall_locs,
            door_ys=hole_locs,
            img_size=self.config.img_size,
            wall_width=self.config.wall_width,
            door_space=self.config.door_space,
            border_wall_loc=self.config.border_wall_loc,
            device=self.device,
        )

    # ------------------------------------------------------------------
    # Physics — check collision against all walls
    # ------------------------------------------------------------------

    def generate_transitions(self, location, actions, bias_angle, walls):
        """Simulate a trajectory, enforcing collisions with all walls.

        Args:
            location:   (1, 2) start position
            actions:    (1, n_steps-1, 2) action sequence
            bias_angle: (1, 2) unused here (kept for API compatibility)
            walls:      tuple(wall_xs (num_walls,), door_ys (num_walls,))

        Returns:
            WallSample with states (1, 2, sample_length, H, W),
            actions (1, 2, sample_length), locations (1, 2, sample_length),
            wall_x (num_walls,), door_y (num_walls,)
        """
        wall_xs, door_ys = walls

        locations = [location]
        for i in range(actions.shape[1]):
            next_location = self.generate_transition(locations[-1], actions[:, i])
            next_location = check_multi_wall_step(
                locations[-1], next_location, wall_xs, door_ys, self.config
            )
            locations.append(next_location)

        # Stack locations: (1, T, 1, 2)
        locations = torch.stack(locations, dim=1).unsqueeze(dim=-2)
        actions = actions.unsqueeze(dim=-2)

        # Render dot channel
        states = self.render_location(locations)  # (1, T, H, W)

        # Render walls channel and broadcast over time
        walls_img = self.render_walls(wall_xs, door_ys)  # (H, W)
        walls_img = walls_img.unsqueeze(0).unsqueeze(0).unsqueeze(0)  # (1, 1, 1, H, W)
        walls_img = walls_img.expand(1, states.shape[1], 1, *walls_img.shape[-2:])

        # Concatenate channels: (1, T, 2, H, W)
        states_with_walls = torch.cat([states, walls_img], dim=-3)

        # Optional temporal downsampling
        if self.config.n_steps_reduce_factor > 1:
            f = self.config.n_steps_reduce_factor
            states_with_walls = states_with_walls[:, ::f]
            locations = locations[:, ::f]
            reduced_chunks = actions.shape[1] // f
            action_chunks = torch.chunk(actions, chunks=reduced_chunks, dim=1)
            actions = torch.cat(
                [torch.sum(chunk, dim=1, keepdim=True) for chunk in action_chunks], dim=1
            )

        # Drop agent dimension
        locations = locations.squeeze(2)       # (1, T, 2)
        actions = actions.squeeze(2)           # (1, T-1, 2)
        states_with_walls = states_with_walls.float()

        # Normalize
        if self.config.normalize:
            states_with_walls = self.normalizer.normalize_state(states_with_walls)
            locations = self.normalizer.normalize_location(locations)

        # Reshape to (B, C, T, H, W)
        states_with_walls = states_with_walls.permute(0, 2, 1, 3, 4)
        states_with_walls = states_with_walls[:, :, :-1]  # drop last timestep

        # (B, 2, T)
        actions = actions.permute(0, 2, 1)
        locations = locations.permute(0, 2, 1)[:, :, :-1]

        # Random subsequence of length sample_length
        max_start = max(1, self.config.n_steps - self.config.sample_length)
        start = np.random.randint(0, max_start)
        sl = self.config.sample_length
        states_with_walls = states_with_walls[:, :, start : start + sl]
        actions = actions[:, :, start : start + sl]
        locations = locations[:, :, start : start + sl]

        return WallSample(
            states=states_with_walls,
            actions=actions,
            locations=locations,
            wall_x=wall_xs,
            door_y=door_ys,
        )
