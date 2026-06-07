"""
Tests for user credential setup utilities.

This file validates user-local credential setup without storing secret values
in the runtime context.

Pytest automatically discovers and runs functions whose names start with
``test_``. Therefore, no explicit test decorator is needed here.
"""

import os
from pathlib import Path

import pytest

from Project_1_SepFormer_TSE_HF.idea_1_project_setup.src.Setup.user_credential import (
    export_environment_variables,
    get_nested_credential_value,
    resolve_credential_path,
    setup_user_credential,
)


#############################################
# Disabled setup
#############################################


def test_setup_user_credential_returns_disabled_context_by_default() -> None:
    context = setup_user_credential()

    assert context == {
        "enabled": False,
        "path": None,
        "loaded": False,
        "exported_env_vars": [],
    }


#############################################
# Path handling
#############################################


def test_resolve_credential_path_rejects_missing_path_when_strict() -> None:
    with pytest.raises(RuntimeError):
        resolve_credential_path(
            path=None,
            strict=True,
        )


#############################################
# Nested credential retrieval
#############################################


def test_get_nested_credential_value_returns_nested_value() -> None:
    credentials = {
        "wandb": {
            "api_key": "dummy_key",
        },
    }

    credential_value = get_nested_credential_value(
        credentials=credentials,
        key_path=["wandb", "api_key"],
    )

    assert credential_value == "dummy_key"


def test_get_nested_credential_value_rejects_missing_key() -> None:
    credentials = {
        "wandb": {},
    }

    with pytest.raises(RuntimeError):
        get_nested_credential_value(
            credentials=credentials,
            key_path=["wandb", "api_key"],
            strict=True,
        )


#############################################
# Environment variable export
#############################################


def test_export_environment_variables_sets_expected_env_var(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("WANDB_API_KEY", raising=False)

    credentials = {
        "wandb": {
            "api_key": "dummy_key",
        },
    }

    exported_env_vars = export_environment_variables(
        credentials=credentials,
        environment_variables={
            "WANDB_API_KEY": ["wandb", "api_key"],
        },
        strict=True,
    )

    assert exported_env_vars == ["WANDB_API_KEY"]
    assert os.environ["WANDB_API_KEY"] == "dummy_key"


def test_setup_user_credential_exports_env_var_without_storing_secret(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("WANDB_API_KEY", raising=False)

    credential_path = tmp_path / "user_credential_local.yaml"
    credential_path.write_text(
        "wandb:\n"
        "  api_key: dummy_key\n",
        encoding="utf-8",
    )

    context = setup_user_credential(
        user_credential_config={
            "enabled": True,
            "strict": True,
            "path": str(credential_path),
            "environment_variables": {
                "WANDB_API_KEY": ["wandb", "api_key"],
            },
        },
    )

    assert os.environ["WANDB_API_KEY"] == "dummy_key"
    assert context == {
        "enabled": True,
        "path": str(credential_path.resolve()),
        "loaded": True,
        "exported_env_vars": ["WANDB_API_KEY"],
    }
    assert "dummy_key" not in str(context)