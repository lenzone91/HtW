import torch


def render_walls_multi(
    wall_xs: torch.Tensor,
    door_ys: torch.Tensor,
    img_size: int,
    wall_width: int,
    door_space: int,
    border_wall_loc: int,
    device: torch.device,
) -> torch.Tensor:
    """Render N vertical walls with doors onto a single (img_size, img_size) canvas.

    Each wall is a vertical stripe at x=wall_xs[i], with a gap (door) centered
    at y=door_ys[i]. Border walls are added on all four sides.

    Args:
        wall_xs:        (num_walls,) x-coordinates of wall centers
        door_ys:        (num_walls,) y-coordinates of door centers
        img_size:       image side length in pixels
        wall_width:     pixel width of each wall stripe
        door_space:     half-size of the door opening (door spans door_space*2 pixels)
        border_wall_loc: thickness of the border walls in pixels
        device:         torch device

    Returns:
        canvas: uint8 tensor of shape (img_size, img_size), values 0 or 255
    """
    x = torch.arange(0, img_size, device=device)
    y = torch.arange(0, img_size, device=device)
    grid_x, grid_y = torch.meshgrid(x, y, indexing="xy")  # (img_size, img_size)

    canvas = torch.zeros(img_size, img_size, dtype=torch.uint8, device=device)
    offset = wall_width // 2

    for wall_x_val, door_y_val in zip(wall_xs.tolist(), door_ys.tolist()):
        in_wall_x = (grid_x >= wall_x_val - offset) & (grid_x <= wall_x_val + offset)
        not_in_door = (
            (grid_y < door_y_val - door_space) | (grid_y > door_y_val + door_space)
        )
        wall_pixels = in_wall_x & not_in_door
        canvas = torch.max(canvas, wall_pixels.to(torch.uint8) * 255)

    # Border walls
    b = border_wall_loc
    canvas[:b, :] = 255
    canvas[-b:, :] = 255
    canvas[:, :b] = 255
    canvas[:, -b:] = 255

    return canvas
