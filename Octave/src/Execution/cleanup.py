def close_external_services() -> None:
    close_wandb_if_active()


def close_wandb_if_active() -> None:
    try:
        import wandb
    except ImportError:
        return

    if getattr(wandb, "run", None) is not None:
        wandb.finish()
