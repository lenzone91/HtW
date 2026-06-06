import argparse
from pathlib import Path

from Project_1_SepFormer_TSE_HF.idea_1_project_setup.src.Configs.conversion import (
    load_config,
)
from Project_1_SepFormer_TSE_HF.idea_1_project_setup.src.Configs.resolve import (
    resolve_config,
)
from Project_1_SepFormer_TSE_HF.idea_1_project_setup.src.Execution.evaluate import (
    run_evaluation,
)
from Project_1_SepFormer_TSE_HF.idea_1_project_setup.src.Execution.train import (
    run_training,
)
from Project_1_SepFormer_TSE_HF.idea_1_project_setup.src.Sweeps.wandb_sweep import (
    run_existing_wandb_sweep,
    run_wandb_sweep,
)

from Project_1_SepFormer_TSE_HF.idea_1_project_setup.src.Setup.project_root import (
    infer_project_root,
)


def main() -> None:
    """
    Resolve a user config and dispatch it to the appropriate execution workflow.
    """
    args = _parse_args()

    input_path = Path(args.config_path).expanduser().resolve()
    resolved_config_path = resolve_config(input_path)
    config = load_config(resolved_config_path)

    mode = _infer_launch_mode(input_path=input_path)

    if mode == "train":
        run_training(
            config=config,
            config_path=resolved_config_path,
        )
        return

    if mode == "validate":
        run_evaluation(
            config=config,
            evaluation_step="validate",
            config_path=resolved_config_path,
        )
        return

    if mode == "sweep":
        _run_sweep(config)
        return

    raise ValueError(f"Unsupported launch mode: {mode}.")

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Launch a training run, validation run, or sweep from a config.",
    )

    parser.add_argument(
        "config_path",
        type=str,
        help="Path to a resolved config file or elementary config folder.",
    )

    return parser.parse_args()


def _infer_launch_mode(input_path: Path) -> str:
    """
    Infer launch mode from the config folder convention.
    """
    path_parts = set(input_path.parts)

    if "runs" in path_parts:
        return "train"

    if "validations" in path_parts:
        return "validate"

    if "sweeps" in path_parts:
        return "sweep"

    raise ValueError(
        "Could not infer launch mode from config path. "
        "Expected path to contain one of: runs, validations, sweeps."
    )


def _run_sweep(config: dict) -> None:
    """
    Dispatch to a new or existing WandB sweep.
    """
    if "sweep_id" in config:
        run_existing_wandb_sweep(config)
        return

    run_wandb_sweep(config)


if __name__ == "__main__":
    main()