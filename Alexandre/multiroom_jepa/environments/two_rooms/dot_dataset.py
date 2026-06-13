import math
from abc import abstractmethod
from dataclasses import dataclass
from typing import NamedTuple

import torch
from scipy.stats import truncnorm

from .normalizer import Normalizer
from .utils import *


class Sample(NamedTuple):
    states: torch.Tensor  # [(batch_size), T, 1, 28, 28]
    locations: torch.Tensor  # [(batch_size), T, N_DOTS, 2]
    actions: torch.Tensor  # [(batch_size), T, 2]
    bias_angle: torch.Tensor  # [(batch_size), 2]


@dataclass
class DotDatasetConfig:
    size: int = 10000
    val_size: int = 10000
    batch_size: int = 128
    dot_std: float = 1.3
    action_noise: float = 0.2
    action_angle_noise: float = 0.15
    action_step_mean: float = 1.0
    action_step_std: float = 0.4
    action_lower_bd: float = 0.2
    action_upper_bd: float = 1.8
    max_step: float = 1.0
    n_steps: int = 91
    img_size: int = 64
    train: bool = True
    device: str = "cuda"
    repeat_actions: int = 1
    n_steps_reduce_factor: int = 1
    border_wall_loc: int = 5  # wall located x pixels away from border
    chunked_actions: bool = False
    normalize: bool = True


class DotDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        config: DotDatasetConfig,
    ):
        super().__init__()
        self.config = config

        _requested = self.config.device
        _available = _requested if _requested != "cuda" or torch.cuda.is_available() else "cpu"
        self.device = torch.device(_available)
        self.padding = config.border_wall_loc - 1

        if self.config.normalize:
            self.normalizer = Normalizer()
        else:
            self.normalizer = None

    def __len__(self):
        return self.config.size

    def __getitem__(self, i):
        out = self.generate_multistep_sample()
        return out._replace(
            states=out.states.squeeze(0),
            actions=out.actions.squeeze(0),
            locations=out.locations.squeeze(0),
        )

    def render_location(self, locations: torch.Tensor):
        x = torch.linspace(
            0, self.config.img_size - 1, steps=self.config.img_size, device=self.device
        )
        y = torch.linspace(
            0, self.config.img_size - 1, steps=self.config.img_size, device=self.device
        )
        xx, yy = torch.meshgrid(x, y, indexing="xy")
        c = torch.stack([xx, yy], dim=-1)
        c = c.view(*([1] * (len(locations.shape) - 1)), *c.shape).repeat(
            *locations.shape[:-1], *([1] * len(c.shape))
        )  # repeat the number of times required for locations.
        locations = locations.unsqueeze(-2).unsqueeze(
            -2
        )  # add dims for compatibility with c
        img = (
            (
                torch.exp(
                    -(c - locations).norm(dim=-1).pow(2)
                    / (2 * self.config.dot_std * self.config.dot_std)
                )
                * 255
            )
            .clamp(0, 255)
            .to(torch.uint8)
        )
        return img

    def generate_state_and_actions(
        self,
        wall_locs=None,
        door_locs=None,
        size=None,
        n_steps=17,
    ):
        location = self.generate_state(wall_locs=wall_locs, door_locs=door_locs)
        actions, bias_angle = self.generate_actions(n_steps=n_steps)
        return location, actions, bias_angle

    def generate_state(self, wall_locs=None, door_locs=None, size=None):
        if size is None:
            size = 1

        effective_range = (self.config.img_size - 1) - 2 * self.padding

        location = (
            torch.rand(size=(size, 2), device=self.device) * effective_range
            + self.padding
        )

        left_walls = wall_locs - self.config.wall_width // 2
        right_walls = wall_locs + self.config.wall_width // 2

        door_top = door_locs + self.config.door_space
        door_bot = door_locs - self.config.door_space

        btw_walls = (location[:, 0] >= left_walls) & (location[:, 0] <= right_walls)
        not_btw_doors = (location[:, 1] < door_bot) | (location[:, 1] > door_top)
        inside_walls = btw_walls & not_btw_doors

        min_val = self.config.border_wall_loc - 1
        max_val = self.config.img_size - self.config.border_wall_loc

        if inside_walls.any():
            change_to_ef = (torch.rand(size) < 0.5).to(door_locs.device)
            new_x_left = sample_uniformly_between(
                torch.full((size,), min_val).to(door_locs.device), left_walls
            )
            new_x_right = sample_uniformly_between(
                right_walls, torch.full((size,), max_val).to(door_locs.device)
            )
            location[inside_walls & change_to_ef, 0] = new_x_left[
                inside_walls & change_to_ef
            ]
            location[inside_walls & ~change_to_ef, 0] = new_x_right[
                inside_walls & ~change_to_ef
            ]

        return location

    def sample_walls(self):
        return None

    def generate_multistep_sample(self):
        walls = self.sample_walls()
        start_location, actions, bias_angle = self.generate_state_and_actions(
            wall_locs=walls[0], door_locs=walls[1], n_steps=self.config.n_steps
        )
        bs = start_location.size(0)
        if self.config.dup_traj_rate > 0:
            raise NotImplementedError()

        sample = self.generate_transitions(
            start_location, actions, bias_angle, walls=walls
        )
        return sample

    @abstractmethod
    def generate_transitions(self, location, actions, bias_angle):
        pass

    def generate_transition(self, location, action):
        return location + action

    def generate_actions(self, n_steps, bias_angle=None):
        if bias_angle is None:
            bias = torch.rand(1, device=self.device) * 2 * math.pi
            bias_angle = DotDataset.angle_to_vec(bias)
            bs = 1
        else:
            bias = self.vec_to_angle(bias_angle)
            bs = bias_angle.shape[0]

        concentration = 1 / self.config.action_angle_noise
        von_mises_dist = torch.distributions.VonMises(
            concentration=concentration, loc=0.0
        )
        angles = [bias]
        for i in range(1, n_steps - 1):
            noise = von_mises_dist.sample((bs,)).to(self.device)
            angles.append((angles[-1] + noise).fmod(2 * torch.pi))
        angles = torch.stack(angles, dim=1)

        a = (
            self.config.action_lower_bd - self.config.action_step_mean
        ) / self.config.action_step_std
        b = (
            self.config.action_upper_bd - self.config.action_step_mean
        ) / self.config.action_step_std
        truncated_normal_dist = truncnorm(
            a, b, loc=self.config.action_step_mean, scale=self.config.action_step_std
        )
        samples = truncated_normal_dist.rvs(size=(bs, n_steps - 1))
        steps = torch.tensor(samples, dtype=torch.float32, device=self.device)
        vecs = DotDataset.angle_to_vec(angles)
        actions = vecs * steps.unsqueeze(-1)

        return actions, bias_angle

    @staticmethod
    def angle_to_vec(a):
        return torch.stack([torch.cos(a), torch.sin(a)], dim=-1)

    @staticmethod
    def vec_to_angle(v):
        return torch.atan2(v[:, 1], v[:, 0])

    @staticmethod
    def xy_to_polar(v):
        norm = torch.norm(v, dim=-1, keepdim=True)
        angle = torch.atan2(v[..., 1], v[..., 0])
        angle = angle % (2 * torch.pi)
        return torch.cat((angle.unsqueeze(-1), norm), dim=-1)

    @staticmethod
    def polar_to_xy(polar):
        angle = polar[..., 0]
        norm = polar[..., 1]
        delta_x = norm * torch.cos(angle)
        delta_y = norm * torch.sin(angle)
        return torch.stack((delta_x, delta_y), dim=-1)
