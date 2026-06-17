import sys


#############################################
# Execution cleanup
#############################################

# Cleanup utilities for external services. Logger-agnostic at the public API
# level; logger-specific cleanup is handled internally and defensively.


def close_external_services() -> None:
    """
    Close external services initialized during execution.

    Defensive: does not require optional logger libraries, does not assume a
    specific logger was used, and silently skips uninitialized services.
    """
    close_wandb_if_active()


def close_wandb_if_active() -> None:
    """
    Finish the active Weights & Biases run if wandb is imported and active.
    """
    wandb = sys.modules.get("wandb")

    if wandb is None:
        return

    if getattr(wandb, "run", None) is None:
        return

    wandb.finish()
