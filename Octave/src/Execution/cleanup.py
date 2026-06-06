import sys


#############################################
# Execution cleanup
#############################################

# Cleanup utilities for external services.
#
# This file must stay logger-agnostic at the public API level.
# Logger-specific cleanup is handled internally and defensively.


def close_external_services() -> None:
    """
    Close external services initialized during execution.

    This function is intentionally defensive:
        - it does not require optional logger libraries;
        - it does not assume that a specific logger was used;
        - it silently skips services that were not initialized.
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