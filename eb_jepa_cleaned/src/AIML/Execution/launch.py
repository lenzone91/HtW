"""
Run launcher.

The single entrypoint for starting a run. It composes the config with Hydra
(Workflow/Configs), builds the runtime context (Workflow/Setup), and dispatches
to the train / resume / validate flow chosen by the CLI `--mode` flag or, if
absent, `config["run"]["mode"]`.

It stays generic (no dependency on any experiment): the experiment's concretes
are registered by importing the modules listed in `config["run"]["imports"]`
(e.g. `src.AcVideoJEPA`) — resolved dynamically, not statically, so
AIML never imports a concrete experiment.

CLI:
    python -m src.AIML.Execution.launch <config_dir> \
        [--config-name config] [--mode train|resume|validate] \
        [--ckpt PATH] [--overwrite | --ask-overwrite] [key=value ...]
"""

import argparse
import importlib

from ...Workflow.Configs.compose import load_resolved_config
from ...Workflow.Setup import build_runtime_context
from .resume import run_resume_training
from .train import run_training
from .validate import run_validation

RUN_MODES = ("train", "resume", "validate")


def launch(
    config_dir: str,
    config_name: str = "config",
    mode: str | None = None,
    overrides: list[str] | None = None,
    existing_run_dir_policy: str | None = None,
    checkpoint_path: str | None = None,
) -> dict:
    """
    Compose config, build runtime context, and dispatch to the chosen run mode.
    """
    config = load_resolved_config(config_dir, config_name, overrides=overrides)
    _import_registrations(config)

    if existing_run_dir_policy is not None:
        config.setdefault("setup", {}).setdefault("paths", {})[
            "existing_run_dir_policy"
        ] = existing_run_dir_policy

    runtime_context = build_runtime_context(config.get("setup", {}))

    mode = mode or config.get("run", {}).get("mode", "train")
    return dispatch(
        mode=mode,
        config=config,
        runtime_context=runtime_context,
        checkpoint_path=checkpoint_path,
    )


def dispatch(
    mode: str,
    config: dict,
    runtime_context: dict | None,
    checkpoint_path: str | None = None,
) -> dict:
    """Route to the run flow for `mode`."""
    if mode == "train":
        return run_training(config, runtime_context)
    if mode == "resume":
        return run_resume_training(config, runtime_context, checkpoint_path=checkpoint_path)
    if mode == "validate":
        return run_validation(config, runtime_context)
    raise ValueError(f"Unknown run mode '{mode}'. Expected one of {RUN_MODES}.")


def _import_registrations(config: dict) -> None:
    """Import the modules that register the experiment's concretes."""
    for module_path in config.get("run", {}).get("imports", []):
        importlib.import_module(module_path)


def main(argv: list[str] | None = None) -> dict:
    parser = argparse.ArgumentParser(description="Launch a run (train/resume/validate).")
    parser.add_argument("config_dir", help="Hydra config directory.")
    parser.add_argument("--config-name", default="config")
    parser.add_argument(
        "--mode",
        choices=RUN_MODES,
        default=None,
        help="Run mode; defaults to config['run']['mode'].",
    )
    parser.add_argument("--ckpt", dest="checkpoint_path", default=None)
    policy = parser.add_mutually_exclusive_group()
    policy.add_argument(
        "--overwrite", dest="policy", action="store_const", const="overwrite"
    )
    policy.add_argument(
        "--ask-overwrite", dest="policy", action="store_const", const="ask"
    )
    parser.add_argument(
        "overrides", nargs="*", help="Hydra-style overrides, e.g. trainer.max_steps=100."
    )
    args = parser.parse_args(argv)

    return launch(
        config_dir=args.config_dir,
        config_name=args.config_name,
        mode=args.mode,
        overrides=args.overrides,
        existing_run_dir_policy=args.policy,
        checkpoint_path=args.checkpoint_path,
    )


if __name__ == "__main__":
    main()
