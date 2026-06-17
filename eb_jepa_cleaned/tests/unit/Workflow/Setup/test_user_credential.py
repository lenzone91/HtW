"""
Workflow/Setup user credential loading: exporting secrets (e.g. the wandb API
key) to the environment, and the credential -> wandb login linkage.
"""

import os

import pytest

from src.Workflow.Setup import build_runtime_context
from src.Workflow.Setup.user_credential import setup_user_credential
from src.Workflow.Setup.wandb_setup import setup_wandb

CRED_YAML = "wandb:\n  api_key: secret-key-123\n"


def _write_credentials(tmp_path):
    path = tmp_path / "user_credential.yaml"
    path.write_text(CRED_YAML)
    return path


def test_disabled_is_noop():
    context = setup_user_credential({"enabled": False})
    assert context == {
        "enabled": False,
        "path": None,
        "loaded": False,
        "exported_env_vars": [],
    }


def test_exports_env_var_without_leaking_secret(monkeypatch, tmp_path):
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    path = _write_credentials(tmp_path)

    context = setup_user_credential(
        {
            "enabled": True,
            "path": str(path),
            "environment_variables": {"WANDB_API_KEY": ["wandb", "api_key"]},
        }
    )

    assert os.environ["WANDB_API_KEY"] == "secret-key-123"
    assert context["loaded"] is True
    assert context["exported_env_vars"] == ["WANDB_API_KEY"]
    # The secret value is never stored in the context.
    assert "secret-key-123" not in str(context)


def test_enabled_without_path_raises():
    with pytest.raises(ValueError):
        setup_user_credential({"enabled": True})


def test_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        setup_user_credential(
            {"enabled": True, "path": str(tmp_path / "nope.yaml")}
        )


def test_missing_key_path_raises(monkeypatch, tmp_path):
    monkeypatch.delenv("MISSING", raising=False)
    path = _write_credentials(tmp_path)
    with pytest.raises(KeyError):
        setup_user_credential(
            {
                "enabled": True,
                "path": str(path),
                "environment_variables": {"MISSING": ["wandb", "absent"]},
            }
        )


def test_runtime_context_settles_wandb_key_then_login(monkeypatch, tmp_path):
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    monkeypatch.delenv("WANDB_MODE", raising=False)
    # Avoid any network: stub the wandb login the setup would call.
    monkeypatch.setattr("wandb.login", lambda **kwargs: True)
    path = _write_credentials(tmp_path)

    runtime_context = build_runtime_context(
        {
            "device": "cpu",
            "paths": {
                "project_root": str(tmp_path),
                "run_name": "r0",
                "existing_run_dir_policy": "overwrite",
            },
            "user_credential": {
                "enabled": True,
                "path": str(path),
                "environment_variables": {"WANDB_API_KEY": ["wandb", "api_key"]},
            },
            "wandb": {"enabled": True, "mode": "online", "login": True},
        }
    )

    # Credential step exported the key before wandb login ran.
    assert os.environ["WANDB_API_KEY"] == "secret-key-123"
    assert runtime_context["credentials"]["exported_env_vars"] == ["WANDB_API_KEY"]
    assert runtime_context["wandb"] == {
        "enabled": True,
        "mode": "online",
        "logged_in": True,
    }
