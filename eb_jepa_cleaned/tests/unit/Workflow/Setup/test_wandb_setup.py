"""Workflow/Setup wandb registration + its inclusion in the runtime context."""

import os
from pathlib import Path

import pytest

from src.Workflow.Setup import build_runtime_context
from src.Workflow.Setup.wandb_setup import setup_wandb


def test_setup_wandb_disabled_is_noop():
    assert setup_wandb({"enabled": False}) == {"enabled": False}


def test_setup_wandb_offline_sets_mode(monkeypatch):
    monkeypatch.delenv("WANDB_MODE", raising=False)
    summary = setup_wandb({"enabled": True, "mode": "offline"})
    assert summary == {"enabled": True, "mode": "offline", "logged_in": False}
    assert os.environ["WANDB_MODE"] == "offline"


def test_setup_wandb_rejects_unknown_mode():
    with pytest.raises(ValueError):
        setup_wandb({"enabled": True, "mode": "bogus"})


def test_setup_wandb_login_requires_api_key(monkeypatch):
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    with pytest.raises(ValueError):
        setup_wandb({"enabled": True, "mode": "online", "login": True})


def test_runtime_context_includes_wandb_when_configured(monkeypatch, tmp_path):
    monkeypatch.delenv("WANDB_MODE", raising=False)
    rc = build_runtime_context(
        {
            "device": "cpu",
            "paths": {
                "project_root": str(tmp_path),
                "run_name": "r0",
                "existing_run_dir_policy": "overwrite",
            },
            "wandb": {"enabled": True, "mode": "offline"},
        }
    )
    assert rc["wandb"] == {"enabled": True, "mode": "offline", "logged_in": False}
    assert Path(rc["paths"]["run_dir"]).is_dir()


def test_runtime_context_omits_wandb_when_absent(tmp_path):
    rc = build_runtime_context(
        {
            "device": "cpu",
            "paths": {
                "project_root": str(tmp_path),
                "run_name": "r1",
                "existing_run_dir_policy": "overwrite",
            },
        }
    )
    assert "wandb" not in rc
