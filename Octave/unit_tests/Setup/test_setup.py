"""
Tests for the high-level runtime setup orchestrator.

This file validates that setup_runtime calls setup substeps coherently and
returns a complete runtime context.

Pytest automatically discovers and runs functions whose names start with
``test_``. Therefore, no explicit test decorator is needed here.

The monkeypatch fixture is used to temporarily replace environment variables
and WandB login behavior during tests. This avoids real external authentication
and keeps the tests isolated.
"""

from pathlib import Path

from Project_1_SepFormer_TSE_HF.idea_1_project_setup.src.Setup import (
    logger_registration,
)
from Project_1_SepFormer_TSE_HF.idea_1_project_setup.src.Setup.setup import (
    setup_runtime,
)


#############################################
# Helpers
#############################################


def make_minimal_project_root(tmp_path: Path) -> Path:
    project_root = tmp_path / "project"
    (project_root / "src").mkdir(parents=True)
    (project_root / "configs").mkdir(parents=True)

    return project_root


def make_minimal_setup_config(tmp_path: Path) -> dict:
    project_root = make_minimal_project_root(tmp_path)

    credential_path = project_root / "user_credential_local.yaml"
    credential_path.write_text(
        "wandb:\n"
        "  api_key: dummy_key\n",
        encoding="utf-8",
    )

    dataset_root = project_root / "dataset"
    dataset_root.mkdir()

    return {
        "environment": {
            "required_imports": ["math"],
        },
        "hardware": {
            "strict": True,
            "accelerator": "cpu",
            "devices": None,
            "require_cuda": False,
        },
        "reproducibility": {
            "seed": 42,
            "deterministic": False,
            "cudnn_benchmark": False,
        },
        "paths": {
            "project_root": str(project_root),
            "run_root": "results",
            "experiment_name": "experiment",
            "run_name": "run_0",
            "sweep_name": None,
            "overwrite": False,
        },
        "data": {
            "strict": True,
            "dataset_roots": {
                "toy_dataset": "dataset",
            },
        },
        "user_credential": {
            "enabled": True,
            "strict": True,
            "path": "user_credential_local.yaml",
            "environment_variables": {
                "WANDB_API_KEY": ["wandb", "api_key"],
            },
        },
        "logger_registration": {
            "wandb": {
                "enabled": True,
                "login": True,
                "mode": "online",
                "api_key_env_var": "WANDB_API_KEY",
                "strict": True,
            },
        },
    }


#############################################
# Runtime setup
#############################################


def test_setup_runtime_returns_all_context_sections(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.delenv("WANDB_API_KEY", raising=False)

    monkeypatch.setattr(
        logger_registration.wandb,
        "login",
        lambda key=None: True,
    )

    runtime_context = setup_runtime(
        setup_config=make_minimal_setup_config(tmp_path),
    )

    assert set(runtime_context.keys()) == {
        "environment",
        "hardware",
        "reproducibility",
        "paths",
        "data",
        "user_credential",
        "logger_registration",
    }


def test_setup_runtime_calls_user_credential_before_logger_registration(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.delenv("WANDB_API_KEY", raising=False)

    observed_key = {}

    def fake_wandb_login(key=None) -> bool:
        observed_key["value"] = key
        return True

    monkeypatch.setattr(
        logger_registration.wandb,
        "login",
        fake_wandb_login,
    )

    setup_runtime(
        setup_config=make_minimal_setup_config(tmp_path),
    )

    assert observed_key["value"] == "dummy_key"


def test_setup_runtime_returns_serializable_runtime_context(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.delenv("WANDB_API_KEY", raising=False)

    monkeypatch.setattr(
        logger_registration.wandb,
        "login",
        lambda key=None: True,
    )

    runtime_context = setup_runtime(
        setup_config=make_minimal_setup_config(tmp_path),
    )

    assert "dummy_key" not in str(runtime_context)
    assert isinstance(runtime_context["paths"]["run_dir"], str)
    assert isinstance(
        runtime_context["data"]["dataset_roots"]["toy_dataset"],
        str,
    )
    assert isinstance(runtime_context["user_credential"]["path"], str)


def test_setup_runtime_uses_empty_defaults(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        logger_registration.wandb,
        "login",
        lambda key=None: True,
    )

    runtime_context = setup_runtime(
        setup_config={
            "paths": {
                "project_root": str(make_minimal_project_root(tmp_path)),
                "run_root": "results",
                "experiment_name": "experiment",
                "run_name": "run_0",
                "overwrite": False,
            },
            "logger_registration": {},
        },
    )

    assert "environment" in runtime_context
    assert "hardware" in runtime_context
    assert "reproducibility" in runtime_context
    assert "paths" in runtime_context
    assert "logger_registration" in runtime_context