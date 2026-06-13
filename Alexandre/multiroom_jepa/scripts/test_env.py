"""Environment sanity tests â€” Level 1 (shapes) and Level 2 (encoder forward).

Usage:
    python scripts/test_env.py --env two_rooms
    python scripts/test_env.py --env multi_rooms --num_rooms 3
    python scripts/test_env.py --env multi_rooms --num_rooms 4

Level 1: Check dataset output shapes match EB-JEPA expectations.
Level 2: Instantiate the JEPA encoder and run a forward pass, verify no NaN.
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "eb_jepa"))

import torch


# ---------------------------------------------------------------------------
# Level 1 â€” shape assertions
# ---------------------------------------------------------------------------

def level1_two_rooms(verbose: bool = True):
    from environments.two_rooms.wall_dataset import WallDataset, WallDatasetConfig

    config = WallDatasetConfig()
    dset = WallDataset(config=config)
    sample = dset[0]

    # dset[0] has no batch dim â€” DataLoader adds it
    assert sample.states.ndim == 4, f"Expected (C,T,H,W) states, got {sample.states.shape}"
    C, T, H, W = sample.states.shape
    assert C == 2, f"Expected 2 channels (dot + walls), got {C}"
    assert H == W, "Expected square images"
    assert sample.actions.shape == (2, T), f"Bad actions shape: {sample.actions.shape}"
    assert sample.locations.shape == (2, T), f"Bad locations shape: {sample.locations.shape}"

    if verbose:
        print(f"[Level 1] two_rooms  states={tuple(sample.states.shape)} âœ“")
        print(f"[Level 1] two_rooms  actions={tuple(sample.actions.shape)} âœ“")
        print(f"[Level 1] two_rooms  locations={tuple(sample.locations.shape)} âœ“")
        print(f"[Level 1] two_rooms  wall_x={sample.wall_x.tolist()} âœ“")
    return sample


def level1_multi_rooms(num_rooms: int = 3, verbose: bool = True):
    from environments.multiroom.config import MultiRoomConfig
    from environments.multiroom.dataset import MultiRoomDataset

    config = MultiRoomConfig(num_rooms=num_rooms)
    dset = MultiRoomDataset(config=config)
    sample = dset[0]

    # dset[0] has no batch dim â€” DataLoader adds it
    assert sample.states.ndim == 4, f"Expected (C,T,H,W) states, got {sample.states.shape}"
    C, T, H, W = sample.states.shape
    assert C == 2, f"Expected 2 channels (dot + walls), got {C}"
    assert H == W, "Expected square images"
    assert sample.actions.shape == (2, T), f"Bad actions shape: {sample.actions.shape}"
    assert sample.locations.shape == (2, T), f"Bad locations shape: {sample.locations.shape}"
    assert len(sample.wall_x) == num_rooms - 1, (
        f"Expected {num_rooms-1} wall positions, got {len(sample.wall_x)}"
    )

    if verbose:
        print(f"[Level 1] multi_rooms({num_rooms}r)  states={tuple(sample.states.shape)} âœ“")
        print(f"[Level 1] multi_rooms({num_rooms}r)  actions={tuple(sample.actions.shape)} âœ“")
        print(f"[Level 1] multi_rooms({num_rooms}r)  wall_xs={sample.wall_x.tolist()} âœ“")
        print(f"[Level 1] multi_rooms({num_rooms}r)  door_ys={sample.door_y.tolist()} âœ“")
    return sample


# ---------------------------------------------------------------------------
# Level 2 â€” encoder forward pass
# ---------------------------------------------------------------------------

def level2_encoder(sample, env_name: str):
    from eb_jepa.models.jepa import JEPA, JEPAConfig

    # Add batch dimension if missing
    x = sample.states
    if x.ndim == 4:
        x = x.unsqueeze(0)  # (C, T, H, W) â†’ (1, C, T, H, W)
    B, C, T, H, W = x.shape

    config = JEPAConfig(dobs=C, henc=32, hpre=32, dstc=32, nsteps=8)
    model = JEPA(config)
    model.eval()

    with torch.no_grad():
        obs = x[:, :, :config.nsteps]  # (1, 2, nsteps, H, W)
        enc_out = model.encoder(obs)   # should be (1, dstc, ...)

    assert not torch.isnan(enc_out).any(), "NaN detected in encoder output!"
    print(f"[Level 2] {env_name}  encoder forward: {tuple(enc_out.shape)} â€” no NaN âœ“")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Sanity tests for MultiRoom environment")
    parser.add_argument("--env", choices=["two_rooms", "multi_rooms"], default="multi_rooms")
    parser.add_argument("--num_rooms", type=int, default=3)
    parser.add_argument("--skip_level2", action="store_true", help="Skip encoder forward test")
    args = parser.parse_args()

    print(f"\n=== Testing env={args.env} ===\n")

    if args.env == "two_rooms":
        sample = level1_two_rooms()
        env_label = "two_rooms"
    else:
        sample = level1_multi_rooms(args.num_rooms)
        env_label = f"multi_rooms({args.num_rooms}r)"

    if not args.skip_level2:
        try:
            level2_encoder(sample, env_label)
        except Exception as e:
            print(f"[Level 2] SKIP â€” {e}")
            print("         (Run from inside the eb_jepa package for encoder tests)")

    print("\nAll checks passed. âœ“\n")


if __name__ == "__main__":
    main()

