"""
Tests for execution cleanup utilities.

This file validates defensive cleanup of optional external services.

Pytest automatically discovers and runs functions whose names start with
``test_``. Therefore, no explicit test decorator is needed here.

The monkeypatch fixture is used to temporarily modify sys.modules during tests.
This avoids depending on whether WandB is actually installed or imported.
"""

import sys

from Project_1_SepFormer_TSE_HF.idea_1_project_setup.src.Execution.cleanup import (
    close_external_services,
    close_wandb_if_active,
)


#############################################
# Dummy objects
#############################################


class DummyWandB:
    def __init__(self, run=None) -> None:
        self.run = run
        self.finish_called = False

    def finish(self) -> None:
        self.finish_called = True


#############################################
# WandB cleanup
#############################################


def test_close_wandb_if_active_does_nothing_when_wandb_not_imported(
    monkeypatch,
) -> None:
    monkeypatch.delitem(sys.modules, "wandb", raising=False)

    close_wandb_if_active()


def test_close_wandb_if_active_does_nothing_when_wandb_has_no_active_run(
    monkeypatch,
) -> None:
    dummy_wandb = DummyWandB(run=None)
    monkeypatch.setitem(sys.modules, "wandb", dummy_wandb)

    close_wandb_if_active()

    assert dummy_wandb.finish_called is False


def test_close_wandb_if_active_finishes_active_wandb_run(
    monkeypatch,
) -> None:
    dummy_wandb = DummyWandB(run=object())
    monkeypatch.setitem(sys.modules, "wandb", dummy_wandb)

    close_wandb_if_active()

    assert dummy_wandb.finish_called is True


#############################################
# Public cleanup
#############################################


def test_close_external_services_calls_wandb_cleanup(monkeypatch) -> None:
    dummy_wandb = DummyWandB(run=object())
    monkeypatch.setitem(sys.modules, "wandb", dummy_wandb)

    close_external_services()

    assert dummy_wandb.finish_called is True