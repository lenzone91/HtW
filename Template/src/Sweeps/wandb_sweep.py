from ..Execution.train import run_training
from .wandb_factory import build_wandb_trial_config


def run_wandb_sweep(config: dict) -> None:
    import wandb

    sweep_id = wandb.sweep(
        sweep=config["wandb_sweep"],
        project=config["agent"].get("project"),
        entity=config["agent"].get("entity"),
    )

    wandb.agent(
        sweep_id=sweep_id,
        function=lambda: run_wandb_trial(config),
        count=config["agent"].get("count"),
        project=config["agent"].get("project"),
        entity=config["agent"].get("entity"),
    )


def run_existing_wandb_sweep(config: dict) -> None:
    import wandb

    wandb.agent(
        sweep_id=config["sweep_id"],
        function=lambda: run_wandb_trial(config),
        count=config["agent"].get("count"),
        project=config["agent"].get("project"),
        entity=config["agent"].get("entity"),
    )


def run_wandb_trial(config: dict) -> dict:
    import wandb

    with wandb.init() as run:
        trial_config = build_wandb_trial_config(
            base_config=config["base_config"],
            sampled_parameters=dict(run.config),
            run_name=run.name,
        )

        return run_training(trial_config)