from __future__ import annotations

import argparse
from pathlib import Path

from src.Execution.train import run_training
from src.Execution.validate import run_validation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Launch an Octave run from a YAML/JSON config."
    )

    parser.add_argument(
        "config_path",
        type=Path,
        help="Path to the run config file.",
    )

    parser.add_argument(
        "--mode",
        choices=("train", "validate"),
        default="train",
        help="Execution mode. Default: train.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = args.config_path.expanduser().resolve()

    if not config_path.is_file():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    if args.mode == "train":
        run_training(config_path=config_path)
        return

    if args.mode == "validate":
        run_validation(config_path=config_path)
        return

    raise ValueError(f"Unsupported execution mode: {args.mode}")


if __name__ == "__main__":
    main()