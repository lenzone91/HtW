"""
Tests for WandB sweep orchestration utilities.

This file validates WandB sweep orchestration without making real WandB calls.

Pytest automatically discovers and runs functions whose names start with
``test_``. Therefore, no explicit test decorator is needed here.

The monkeypatch fixture is used to temporarily replace the WandB module,
trial-config builder, and training entry point. This isolates Sweeps/
orchestration from WandB services and Execution/ training logic.
"""

import sys


from Project_1_SepFormer_TSE_HF.idea_1_project_setup.src.Sweeps import (
    wandb_sweep,
)
from Project_1_SepFormer_TSE_HF.idea_1_project_setup.src.Sweeps.wandb_sweep import (
    run_existing_wandb_sweep,
    run_wandb_sweep,
    run_wandb_trial,
)


#############################################
# Dummy objects
#############################################


class DummyWandBRun:
    def __init__(self) -> None:
        self.config = {"trainer.max_epochs": 5}
        self.name = "dummy_wandb_run"

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        return None


class DummyWandB:
    def __init__(self) -> None:
        self.sweep_called_with = None
        self.agent_called_with = None
        self.init_called = False

    def sweep(self, sweep, project=None, entity=None):
        self.sweep_called_with = {
            "sweep": sweep,
            "project": project,
            "entity": entity,
        }
        return "dummy_sweep_id"

    def agent(self, sweep_id, function, count=None, project=None, entity=None):
        self.agent_called_with = {
            "sweep_id": sweep_id,
            "function": function,
            "count": count,
            "project": project,
            "entity": entity,
        }

    def init(self):
        self.init_called = True
        return DummyWandBRun()


#############################################
# Helpers
#############################################


def make_sweep_config() -> dict:
    return {
        "base_config": {"setup": {"run_name": "base_run"}},
        "wandb_sweep": {"method": "grid"},
        "sweep_id": "existing_sweep_id",
        "agent": {
            "project": "dummy_project",
            "entity": "dummy_entity",
            "count": 1,
        },
    }


#############################################
# Trial execution
#############################################


def test_run_wandb_trial_builds_trial_config_and_runs_training(monkeypatch) -> None:
    dummy_wandb = DummyWandB()
    observed = {}

    monkeypatch.setitem(sys.modules, "wandb", dummy_wandb)

    def fake_build_wandb_trial_config(
        base_config,
        sampled_parameters,
        run_name=None,
    ):
        observed["base_config"] = base_config
        observed["sampled_parameters"] = sampled_parameters
        observed["run_name"] = run_name
        return {"trial": "config"}

    def fake_run_training(config):
        observed["training_config"] = config
        return {"status": "finished"}

    monkeypatch.setattr(
        wandb_sweep,
        "build_wandb_trial_config",
        fake_build_wandb_trial_config,
    )
    monkeypatch.setattr(
        wandb_sweep,
        "run_training",
        fake_run_training,
    )

    report = run_wandb_trial(make_sweep_config())

    assert dummy_wandb.init_called is True
    assert observed["base_config"] == {"setup": {"run_name": "base_run"}}
    assert observed["sampled_parameters"] == {"trainer.max_epochs": 5}
    assert observed["run_name"] == "dummy_wandb_run"
    assert observed["training_config"] == {"trial": "config"}
    assert report == {"status": "finished"}


#############################################
# Sweep orchestration
#############################################


def test_run_wandb_sweep_creates_sweep_and_launches_agent(monkeypatch) -> None:
    dummy_wandb = DummyWandB()
    monkeypatch.setitem(sys.modules, "wandb", dummy_wandb)

    run_wandb_sweep(make_sweep_config())

    assert dummy_wandb.sweep_called_with == {
        "sweep": {"method": "grid"},
        "project": "dummy_project",
        "entity": "dummy_entity",
    }

    assert dummy_wandb.agent_called_with["sweep_id"] == "dummy_sweep_id"
    assert dummy_wandb.agent_called_with["count"] == 1
    assert dummy_wandb.agent_called_with["project"] == "dummy_project"
    assert dummy_wandb.agent_called_with["entity"] == "dummy_entity"
    assert callable(dummy_wandb.agent_called_with["function"])


def test_run_existing_wandb_sweep_launches_agent(monkeypatch) -> None:
    dummy_wandb = DummyWandB()
    monkeypatch.setitem(sys.modules, "wandb", dummy_wandb)

    run_existing_wandb_sweep(make_sweep_config())

    assert dummy_wandb.agent_called_with["sweep_id"] == "existing_sweep_id"
    assert dummy_wandb.agent_called_with["count"] == 1
    assert dummy_wandb.agent_called_with["project"] == "dummy_project"
    assert dummy_wandb.agent_called_with["entity"] == "dummy_entity"
    assert callable(dummy_wandb.agent_called_with["function"])