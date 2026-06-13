from __future__ import annotations

import argparse
from pathlib import Path

try:
    from Octave.src.Execution.train import run_training
    from Octave.src.Execution.validate import run_validation
    from Octave.src.Setup.config_resolution import load_config
except ModuleNotFoundError:
    from src.Execution.train import run_training
    from src.Execution.validate import run_validation
    from src.Setup.config_resolution import load_config


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
    run_dir_policy_group = parser.add_mutually_exclusive_group()
    run_dir_policy_group.add_argument(
        "--overwrite",
        action="store_true",
        help="Override setup.paths.existing_run_dir_policy to 'overwrite'.",
    )
    run_dir_policy_group.add_argument(
        "--ask-overwrite",
        action="store_true",
        help="Override setup.paths.existing_run_dir_policy to 'ask'.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = args.config_path.expanduser().resolve()

    if not config_path.is_file():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    config = load_config(config_path)
    apply_existing_run_dir_policy_override(
        config=config,
        overwrite=args.overwrite,
        ask_overwrite=args.ask_overwrite,
    )

    if args.mode == "train":
        run_training(config=config, config_path=config_path)
        return

    if args.mode == "validate":
        run_validation(config=config, config_path=config_path)
        return

    raise ValueError(f"Unsupported execution mode: {args.mode}")


def apply_existing_run_dir_policy_override(
    config: dict,
    overwrite: bool = False,
    ask_overwrite: bool = False,
) -> None:
    if not overwrite and not ask_overwrite:
        return

    config.setdefault("setup", {})
    config["setup"].setdefault("paths", {})

    if overwrite:
        config["setup"]["paths"]["existing_run_dir_policy"] = "overwrite"
        return

    if ask_overwrite:
        config["setup"]["paths"]["existing_run_dir_policy"] = "ask"


if __name__ == "__main__":
    main()
