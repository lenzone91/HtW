"""Generate and persist dataset samples for inspection.

Usage:
    # MultiRoom — 3 rooms, 5 samples
    python scripts/generate_sample.py --env multi_rooms --num_rooms 3 --n_samples 5

    # TwoRooms — 5 samples
    python scripts/generate_sample.py --env two_rooms --n_samples 5

    # Force the dot to cross all walls (guaranteed, with retries)
    python scripts/generate_sample.py --env multi_rooms --num_rooms 3 --force_cross

    # Also save raw .pt tensors (off by default)
    python scripts/generate_sample.py --env multi_rooms --save_pt

    # Longer trajectories for better visualization
    python scripts/generate_sample.py --env two_rooms --n_frames 80

Config loaded from data_config.yaml — same params as EB-JEPA training.
--force_cross sets cross_wall_rate=1.0 and retries until the dot genuinely crosses.
"""

import argparse
import dataclasses
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "eb_jepa"))

import numpy as np
import torch
import yaml
import imageio

from environments.two_rooms.utils import update_config_from_yaml


# ---------------------------------------------------------------------------
# Config loaders
# ---------------------------------------------------------------------------

def load_two_rooms_config(n_frames=None, force_cross=False):
    from environments.two_rooms.wall_dataset import WallDatasetConfig

    cfg_path = ROOT / "environments" / "two_rooms" / "data_config.yaml"
    with open(cfg_path) as f:
        raw = yaml.safe_load(f)
    raw["device"] = "cpu"
    raw["normalize"] = False   # keep pixel coords for crossing detection + visualization
    if n_frames is not None:
        raw["sample_length"] = n_frames
    if force_cross:
        raw["cross_wall_rate"] = 1.0
        # No random subsampling: generate exactly sample_length frames
        raw["n_steps"] = raw["sample_length"] + 1
    return update_config_from_yaml(WallDatasetConfig, raw)


def load_multiroom_config(num_rooms: int, n_frames=None, force_cross=False):
    from environments.multiroom.config import MultiRoomConfig

    cfg_path = ROOT / "environments" / "multiroom" / "data_config.yaml"
    with open(cfg_path) as f:
        raw = yaml.safe_load(f)
    raw["device"] = "cpu"
    raw["normalize"] = False
    raw["num_rooms"] = num_rooms
    if n_frames is not None:
        raw["sample_length"] = n_frames
    if force_cross:
        raw["cross_wall_rate"] = 1.0
        # No random subsampling: generate exactly sample_length frames
        raw["n_steps"] = raw["sample_length"] + 1
    return update_config_from_yaml(MultiRoomConfig, raw)


# ---------------------------------------------------------------------------
# Crossing detection (pixel-space locations — normalize=False required)
# ---------------------------------------------------------------------------

def _did_cross_two_rooms(sample) -> bool:
    x_traj = sample.locations[0]     # (T,) — x positions in pixels
    wall_x = sample.wall_x[0].item()
    return (x_traj < wall_x - 1).any().item() and (x_traj > wall_x + 1).any().item()


def _did_cross_multi_rooms(sample) -> bool:
    x_traj = sample.locations[0]     # (T,) — x positions in pixels
    wall_xs = sample.wall_x           # (num_walls,)
    leftmost = wall_xs[0].item()
    rightmost = wall_xs[-1].item()
    return (x_traj < leftmost - 1).any().item() and (x_traj > rightmost + 1).any().item()


def get_sample(dset, force_cross: bool, cross_check_fn, max_retries: int = 60):
    """Draw a sample; if force_cross, retry until the dot genuinely crosses."""
    for attempt in range(max_retries if force_cross else 1):
        sample = dset[0]
        if not force_cross or cross_check_fn(sample):
            return sample, attempt + 1
    # Last resort: return whatever we have
    return sample, max_retries


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def make_composite_frames(states: torch.Tensor) -> list:
    """Convert (C=2, T, H, W) state tensor to a list of T RGB frames.

    Walls in gray, agent dot in red. Upscaled 4× for readability.
    normalize=False means states are uint8 [0, 255] — no re-scaling needed.
    """
    C, T, H, W = states.shape
    SCALE = 4
    frames = []
    for t in range(T):
        dot_ch = states[0, t].numpy().astype(np.float32)
        wall_ch = states[1, t].numpy().astype(np.float32)

        def norm(x):
            lo, hi = x.min(), x.max()
            return (x - lo) / max(float(hi - lo), 1e-8)

        dot_n = norm(dot_ch)
        wall_n = norm(wall_ch)

        wall_u8 = (wall_n * 200).astype(np.uint8)
        rgb = np.stack([wall_u8, wall_u8, wall_u8], axis=-1)

        dot_u8 = (dot_n * 255).astype(np.uint8)
        rgb[:, :, 0] = np.clip(rgb[:, :, 0].astype(np.int32) + dot_u8, 0, 255).astype(np.uint8)
        rgb[:, :, 1] = np.clip(rgb[:, :, 1].astype(np.int32) - dot_u8 // 2, 0, 255).astype(np.uint8)
        rgb[:, :, 2] = np.clip(rgb[:, :, 2].astype(np.int32) - dot_u8 // 2, 0, 255).astype(np.uint8)

        rgb = rgb.repeat(SCALE, axis=0).repeat(SCALE, axis=1)
        frames.append(rgb)
    return frames


def save_sample_gif(states: torch.Tensor, path: Path, fps: int = 12):
    imageio.mimsave(str(path), make_composite_frames(states), fps=fps, loop=0)


# ---------------------------------------------------------------------------
# Generators
# ---------------------------------------------------------------------------

def _next_index(output_dir: Path, tag: str) -> int:
    """Return the next available sample index for *tag* in output_dir."""
    existing = list(output_dir.glob(f"{tag}_sample_*.gif"))
    if not existing:
        return 0
    indices = []
    for p in existing:
        stem = p.stem  # e.g. "two_rooms_sample_007"
        try:
            indices.append(int(stem.rsplit("_", 1)[-1]))
        except ValueError:
            pass
    return max(indices) + 1 if indices else 0


def generate_two_rooms(n_samples, output_dir, save_pt, fps, n_frames, force_cross):
    from environments.two_rooms.wall_dataset import WallDataset

    config = load_two_rooms_config(n_frames=n_frames, force_cross=force_cross)
    dset = WallDataset(config=config)
    tag = "two_rooms"
    start = _next_index(output_dir, tag)

    for i in range(start, start + n_samples):
        sample, tries = get_sample(dset, force_cross, _did_cross_two_rooms)
        states = sample.states
        gif_path = output_dir / f"{tag}_sample_{i:03d}.gif"
        save_sample_gif(states, gif_path, fps=fps)

        crossed = _did_cross_two_rooms(sample)
        cross_str = " [CROSSED]" if crossed else ""
        tries_str = f" ({tries} tries)" if force_cross else ""
        print(
            f"  [{i:03d}] {tuple(states.shape)}"
            f"  wall_x={sample.wall_x.tolist()}  door_y={sample.door_y.tolist()}"
            f"{cross_str}{tries_str}  -> {gif_path.name}"
        )

        if save_pt:
            torch.save(
                {"states": sample.states, "actions": sample.actions,
                 "locations": sample.locations, "wall_x": sample.wall_x,
                 "door_y": sample.door_y},
                output_dir / f"{tag}_sample_{i:03d}.pt",
            )


def generate_multi_rooms(n_samples, num_rooms, output_dir, save_pt, fps, n_frames, force_cross):
    from environments.multiroom.dataset import MultiRoomDataset

    config = load_multiroom_config(num_rooms=num_rooms, n_frames=n_frames, force_cross=force_cross)
    dset = MultiRoomDataset(config=config)
    tag = f"multi_rooms_{num_rooms}r"
    start = _next_index(output_dir, tag)

    for i in range(start, start + n_samples):
        sample, tries = get_sample(dset, force_cross, _did_cross_multi_rooms)
        states = sample.states
        gif_path = output_dir / f"{tag}_sample_{i:03d}.gif"
        save_sample_gif(states, gif_path, fps=fps)

        crossed = _did_cross_multi_rooms(sample)
        cross_str = " [CROSSED]" if crossed else ""
        tries_str = f" ({tries} tries)" if force_cross else ""
        print(
            f"  [{i:03d}] {tuple(states.shape)}"
            f"  wall_xs={sample.wall_x.tolist()}"
            f"  door_ys={sample.door_y.tolist()}"
            f"{cross_str}{tries_str}  -> {gif_path.name}"
        )

        if save_pt:
            torch.save(
                {"states": sample.states, "actions": sample.actions,
                 "locations": sample.locations, "wall_x": sample.wall_x,
                 "door_y": sample.door_y},
                output_dir / f"{tag}_sample_{i:03d}.pt",
            )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Generate dataset samples")
    parser.add_argument("--env", choices=["two_rooms", "multi_rooms"], default="multi_rooms")
    parser.add_argument("--num_rooms", type=int, default=3)
    parser.add_argument("--n_samples", type=int, default=5)
    parser.add_argument("--output_dir", type=str, default="output")
    parser.add_argument("--save_pt", action="store_true")
    parser.add_argument("--fps", type=int, default=12)
    parser.add_argument(
        "--n_frames", type=int, default=None,
        help="Frames per GIF (default: sample_length from yaml = 17). Use e.g. 80 to see full trajectory."
    )
    parser.add_argument(
        "--force_cross", action="store_true",
        help="Guarantee the dot crosses all walls. Retries until it genuinely does."
    )
    args = parser.parse_args()

    output_dir = ROOT / args.output_dir
    output_dir.mkdir(exist_ok=True)

    n_frames = args.n_frames

    # When force_cross, ensure n_frames is large enough to complete the crossing.
    # Expert path length ≈ num_walls * 20 steps (0.9px/step, 11px room width + y variance).
    if args.force_cross:
        num_walls = (args.num_rooms - 1) if args.env == "multi_rooms" else 1
        min_frames = max(40, num_walls * 25)
        if n_frames is None or n_frames < min_frames:
            if n_frames is not None and n_frames < min_frames:
                print(
                    f"  Note: --n_frames {n_frames} too short for force_cross with "
                    f"{args.num_rooms} rooms. Auto-bumping to {min_frames}."
                )
            n_frames = min_frames

    label = f"env={args.env}" + (f" num_rooms={args.num_rooms}" if args.env == "multi_rooms" else "")
    if args.force_cross:
        label += f" [force_cross=ON, n_frames={n_frames}]"
    print(f"Generating {args.n_samples} samples [{label}] -> {output_dir}/")

    if args.env == "two_rooms":
        generate_two_rooms(
            args.n_samples, output_dir, args.save_pt, args.fps, n_frames, args.force_cross
        )
    else:
        generate_multi_rooms(
            args.n_samples, args.num_rooms, output_dir, args.save_pt, args.fps,
            n_frames, args.force_cross
        )

    print("Done.")


if __name__ == "__main__":
    main()
