import random
from dataclasses import dataclass
from typing import Optional

import torch

from environments.two_rooms.wall_dataset import WallDatasetConfig


@dataclass
class MultiRoomConfig(WallDatasetConfig):
    """Configuration for the MultiRoom environment.

    Extends WallDatasetConfig with multi-room parameters.
    All single-wall params (wall_width, door_space, etc.) are inherited
    and apply to every wall uniformly.
    """

    num_rooms: int = 3          # number of rooms (creates num_rooms-1 walls)
    min_room_width: int = 10    # minimum pixel width of each room
    cross_wall_rate: float = 0.35  # fraction of samples with a full left→right crossing


def sample_multiroom_walls(config: MultiRoomConfig, device: torch.device = None):
    """Sample wall and door positions for a MultiRoom layout.

    Places num_rooms-1 walls in sorted order, ensuring each room is at
    least min_room_width pixels wide.

    Args:
        config: MultiRoomConfig
        device: resolved torch device (if None, falls back to CPU when CUDA unavailable)

    Returns:
        wall_xs: FloatTensor of shape (num_rooms-1,) — x positions of walls
        door_ys: FloatTensor of shape (num_rooms-1,) — y positions of door centers
    """
    num_walls = config.num_rooms - 1
    if device is None:
        _cuda_ok = config.device != "cuda" or torch.cuda.is_available()
        device = torch.device(config.device if _cuda_ok else "cpu")

    border = config.border_wall_loc
    img_size = config.img_size
    w = config.wall_width
    min_w = config.min_room_width

    # Sequential placement: place walls left to right
    wall_xs = []
    current_min = border + min_w
    for i in range(num_walls):
        remaining = num_walls - 1 - i
        current_max = img_size - border - min_w - remaining * (min_w + w)
        if current_min >= current_max:
            # Fallback: evenly spaced walls
            total = img_size - 2 * border
            step = total // (num_walls + 1)
            wall_xs = [border + step * (j + 1) for j in range(num_walls)]
            break
        wall_x = random.randint(int(current_min), int(current_max))
        wall_xs.append(wall_x)
        current_min = wall_x + w + min_w

    # Random door y position for each wall
    door_min = border + config.door_space + 1
    door_max = img_size - border - config.door_space - 1
    door_ys = [random.randint(door_min, door_max) for _ in range(num_walls)]

    return (
        torch.tensor(wall_xs, dtype=torch.float32, device=device),
        torch.tensor(door_ys, dtype=torch.float32, device=device),
    )
