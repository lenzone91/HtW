"""
Unit tests for AIML.Execution.Runs.cleanup.
"""

import sys

from eb_jepa_cleaned.AIML.Execution.Runs.cleanup import (
    close_external_services,
)


def test_close_external_services_noop_without_wandb():
    # wandb not imported/active -> defensive no-op (must not raise).
    assert sys.modules.get("wandb") is None or getattr(
        sys.modules["wandb"], "run", None
    ) is None
    close_external_services()
