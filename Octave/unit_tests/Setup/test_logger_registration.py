"""
Tests for logger registration setup utilities.

This file validates WandB registration logic without performing a real WandB
login.

Pytest automatically discovers and runs functions whose names start with
``test_``. Therefore, no explicit test decorator is needed here.
"""


import pytest

from Project_1_SepFormer_TSE_HF.idea_1_project_setup.src.Setup import (
    logger_registration,
)
from Project_1_SepFormer_TSE_HF.idea_1_project_setup.src.Setup.logger_registration import (
    build_wandb_context,
    login_wandb,
    setup_logger_registration,
    setup_one_logger_registration,
    setup_wandb_registration,
)


#############################################
# WandB context
#############################################


def test_build_wandb_context_does_not_store_api_key() -> None:
    context = build_wandb_context(
        enabled=True,
        logged_in=True,
        mode="online",
        api_key_env_var="WANDB_API_KEY",
        api_key_found=True,
    )

    assert context == {
        "enabled": True,
        "logged_in": True,
        "mode": "online",
        "api_key_env_var": "WANDB_API_KEY",
        "api_key_found": True,
    }
    assert "dummy_key" not in str(context)


def test_setup_wandb_registration_returns_disabled_context() -> None:
    context = setup_wandb_registration(
        wandb_config={
            "enabled": False,
            "login": True,
            "mode": "online",
            "api_key_env_var": "WANDB_API_KEY",
            "strict": True,
        },
    )

    assert context == {
        "enabled": False,
        "logged_in": None,
        "mode": "online",
        "api_key_env_var": "WANDB_API_KEY",
        "api_key_found": None,
    }


#############################################
# WandB login
#############################################


def test_login_wandb_rejects_missing_api_key_env_var(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("WANDB_API_KEY", raising=False)

    with pytest.raises(RuntimeError):
        login_wandb(
            api_key_env_var="WANDB_API_KEY",
            strict=True,
        )


def test_login_wandb_calls_wandb_login_with_env_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed_key = {}

    monkeypatch.setenv("WANDB_API_KEY", "dummy_key")

    def fake_wandb_login(key: str | None = None) -> bool:
        observed_key["value"] = key
        return True

    monkeypatch.setattr(
        logger_registration.wandb,
        "login",
        fake_wandb_login,
    )

    logged_in = login_wandb(
        api_key_env_var="WANDB_API_KEY",
        strict=True,
    )

    assert logged_in is True
    assert observed_key["value"] == "dummy_key"


def test_setup_wandb_registration_uses_env_var_and_returns_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("WANDB_API_KEY", "dummy_key")

    monkeypatch.setattr(
        logger_registration.wandb,
        "login",
        lambda key=None: True,
    )

    context = setup_wandb_registration(
        wandb_config={
            "enabled": True,
            "login": True,
            "mode": "online",
            "api_key_env_var": "WANDB_API_KEY",
            "strict": True,
        },
    )

    assert context == {
        "enabled": True,
        "logged_in": True,
        "mode": "online",
        "api_key_env_var": "WANDB_API_KEY",
        "api_key_found": True,
    }
    assert "dummy_key" not in str(context)


#############################################
# Logger dispatcher
#############################################


def test_setup_one_logger_registration_rejects_unknown_logger() -> None:
    with pytest.raises(ValueError):
        setup_one_logger_registration(
            logger_name="unknown_logger",
            logger_config={},
        )


def test_setup_logger_registration_dispatches_configured_loggers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("WANDB_API_KEY", "dummy_key")

    monkeypatch.setattr(
        logger_registration.wandb,
        "login",
        lambda key=None: True,
    )

    context = setup_logger_registration(
        logger_registration_config={
            "wandb": {
                "enabled": True,
                "login": True,
                "mode": "online",
                "api_key_env_var": "WANDB_API_KEY",
                "strict": True,
            },
        },
    )

    assert set(context.keys()) == {"wandb"}
    assert context["wandb"]["logged_in"] is True