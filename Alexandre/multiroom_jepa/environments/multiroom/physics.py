import torch

from environments.two_rooms.utils import check_wall_intersect


def check_multi_wall_step(
    current_loc: torch.Tensor,
    next_loc: torch.Tensor,
    wall_xs: torch.Tensor,
    door_ys: torch.Tensor,
    config,
) -> torch.Tensor:
    """Resolve one physics step against multiple walls and borders.

    For each item in the batch, checks:
    1. Border collision — block if the agent would exit the arena
    2. Each wall in order — block if the agent would cross a wall
       outside of its door

    Uses check_wall_intersect() from two_rooms/utils.py which returns
    non-None when a crossing is blocked (i.e. not through a door).

    Args:
        current_loc: (bs, 2) current agent positions
        next_loc:    (bs, 2) proposed next positions
        wall_xs:     (num_walls,) x-coordinates of wall centers
        door_ys:     (num_walls,) y-coordinates of door centers
        config:      MultiRoomConfig

    Returns:
        result: (bs, 2) resolved next positions (= current_loc if blocked)
    """
    result = next_loc.clone()
    bs = current_loc.shape[0]
    device = current_loc.device

    left_border = float(config.border_wall_loc - 1)
    right_border = float(config.img_size - config.border_wall_loc)
    top_border = left_border
    bot_border = right_border

    wall_xs_list = wall_xs.tolist()
    door_ys_list = door_ys.tolist()

    for i in range(bs):
        pos = current_loc[i]
        npos = next_loc[i]

        # Border check
        border_hit = (
            (torch.sign(pos[0] - left_border) * torch.sign(npos[0] - left_border) <= 0)
            | (torch.sign(pos[0] - right_border) * torch.sign(npos[0] - right_border) <= 0)
            | (torch.sign(pos[1] - top_border) * torch.sign(npos[1] - top_border) <= 0)
            | (torch.sign(pos[1] - bot_border) * torch.sign(npos[1] - bot_border) <= 0)
        )
        if border_hit.item():
            result[i] = current_loc[i]
            continue

        # Wall checks — stop at the first wall that blocks
        for wall_x_val, door_y_val in zip(wall_xs_list, door_ys_list):
            wall_x = torch.tensor(wall_x_val, device=device)
            door_y = torch.tensor(door_y_val, device=device)

            intersect, _ = check_wall_intersect(
                pos, npos,
                wall_x, door_y,
                config.wall_width,
                config.door_space,
                config.border_wall_loc,
                config.img_size,
                add_noise=False,
            )
            if intersect is not None:
                result[i] = current_loc[i]
                break

    return result
