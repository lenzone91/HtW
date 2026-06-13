import os

from Octave.src.Setup import logger_registration
from Octave.src.Setup.setup import setup_runtime


def test_setup_runtime_returns_paths_reproducibility_and_logger_context(
    tmp_path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        logger_registration,
        "setup_wandb_registration",
        lambda config: {"enabled": False, "mode": "disabled"},
    )

    runtime_context = setup_runtime(
        setup_config={
            "paths": {
                "project_root": str(tmp_path),
                "run_root": "runs",
                "experiment_name": "experiment",
                "run_name": "run_0",
                "overwrite": False,
            },
            "reproducibility": {
                "seed": 123,
            },
            "logger_registration": {
                "wandb": {
                    "enabled": False,
                },
            },
        }
    )

    assert set(runtime_context) == {
        "credentials",
        "paths",
        "reproducibility",
        "logger_registration",
    }
    assert runtime_context["credentials"]["loaded_env_vars"] == []
    assert runtime_context["reproducibility"]["seed"] == 123


def test_setup_wandb_registration_sets_disabled_mode(monkeypatch) -> None:
    monkeypatch.delenv("WANDB_MODE", raising=False)

    context = logger_registration.setup_wandb_registration(
        {
            "enabled": False,
        }
    )

    assert context["enabled"] is False
    assert os.environ["WANDB_MODE"] == "disabled"


def test_setup_wandb_registration_can_login(monkeypatch) -> None:
    observed = {}

    class DummyWandb:
        @staticmethod
        def login(key=None):
            observed["key"] = key

    monkeypatch.setenv("WANDB_API_KEY", "dummy")
    monkeypatch.setitem(__import__("sys").modules, "wandb", DummyWandb)

    context = logger_registration.setup_wandb_registration(
        {
            "enabled": True,
            "mode": "offline",
            "login": True,
            "api_key_env_var": "WANDB_API_KEY",
            "strict": True,
        }
    )

    assert context["enabled"] is True
    assert context["login_attempted"] is True
    assert observed["key"] == "dummy"
