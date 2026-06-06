"""
Pytest test module for scripts.launch.

The tests isolate launch.py from real training, evaluation, sweep execution,
config loading, and config resolution. launch.py is tested only as a thin
CLI dispatcher.
"""

from pathlib import Path

import pytest

from Project_1_SepFormer_TSE_HF.idea_1_project_setup.scripts import launch


#############################################
# Launch mode inference tests
#############################################


def test_infer_launch_mode_detects_train_run() -> None:
    mode = launch._infer_launch_mode(Path("configs/runs/toy_run_001"))

    assert mode == "train"


def test_infer_launch_mode_detects_validation_run() -> None:
    mode = launch._infer_launch_mode(Path("configs/validations/eval_001"))

    assert mode == "validate"


def test_infer_launch_mode_detects_sweep() -> None:
    mode = launch._infer_launch_mode(Path("configs/sweeps/toy_sweep_001"))

    assert mode == "sweep"


def test_infer_launch_mode_unknown_path_raises() -> None:
    with pytest.raises(ValueError):
        launch._infer_launch_mode(Path("configs/unknown/toy"))


#############################################
# Main dispatch tests
#############################################


def test_launch_train_path_dispatches_to_run_training(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = []

    def fake_parse_args():
        class Args:
            config_path = "configs/runs/toy_run_001"

        return Args()

    def fake_resolve_config(path: Path) -> Path:
        calls.append(("resolve", path))
        return Path("configs/runs/toy_run_001.yaml")

    def fake_load_config(path: Path) -> dict:
        calls.append(("load", path))
        return {"kind": "train"}

    def fake_run_training(
        config: dict,
        config_path: Path | None = None,
    ) -> None:
        calls.append(("train", config, config_path))

    monkeypatch.setattr(launch, "_parse_args", fake_parse_args)
    monkeypatch.setattr(launch, "resolve_config", fake_resolve_config)
    monkeypatch.setattr(launch, "load_config", fake_load_config)
    monkeypatch.setattr(launch, "run_training", fake_run_training)

    launch.main()

    assert calls == [
        ("resolve", Path("configs/runs/toy_run_001").expanduser().resolve()),
        ("load", Path("configs/runs/toy_run_001.yaml")),
        ("train", {"kind": "train"}, Path("configs/runs/toy_run_001.yaml")),
    ]


def test_launch_validation_path_dispatches_to_run_evaluation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = []

    def fake_parse_args():
        class Args:
            config_path = "configs/validations/eval_001"

        return Args()

    def fake_resolve_config(path: Path) -> Path:
        calls.append(("resolve", path))
        return Path("configs/validations/eval_001.yaml")

    def fake_load_config(path: Path) -> dict:
        calls.append(("load", path))
        return {"kind": "validate"}

    def fake_run_evaluation(
        config: dict,
        evaluation_step: str,
        config_path: Path | None = None,
    ) -> None:
        calls.append(("validate", config, evaluation_step, config_path))

    monkeypatch.setattr(launch, "_parse_args", fake_parse_args)
    monkeypatch.setattr(launch, "resolve_config", fake_resolve_config)
    monkeypatch.setattr(launch, "load_config", fake_load_config)
    monkeypatch.setattr(launch, "run_evaluation", fake_run_evaluation)

    launch.main()

    assert calls == [
        ("resolve", Path("configs/validations/eval_001").expanduser().resolve()),
        ("load", Path("configs/validations/eval_001.yaml")),
        (
            "validate",
            {"kind": "validate"},
            "validate",
            Path("configs/validations/eval_001.yaml"),
        ),
    ]


def test_launch_sweep_path_dispatches_to_run_wandb_sweep(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = []

    def fake_parse_args():
        class Args:
            config_path = "configs/sweeps/toy_sweep_001"

        return Args()

    def fake_resolve_config(path: Path) -> Path:
        calls.append(("resolve", path))
        return Path("configs/sweeps/toy_sweep_001.yaml")

    def fake_load_config(path: Path) -> dict:
        calls.append(("load", path))
        return {"kind": "sweep"}

    def fake_run_wandb_sweep(config: dict) -> None:
        calls.append(("new_sweep", config))

    monkeypatch.setattr(launch, "_parse_args", fake_parse_args)
    monkeypatch.setattr(launch, "resolve_config", fake_resolve_config)
    monkeypatch.setattr(launch, "load_config", fake_load_config)
    monkeypatch.setattr(launch, "run_wandb_sweep", fake_run_wandb_sweep)

    launch.main()

    assert calls == [
        ("resolve", Path("configs/sweeps/toy_sweep_001").expanduser().resolve()),
        ("load", Path("configs/sweeps/toy_sweep_001.yaml")),
        ("new_sweep", {"kind": "sweep"}),
    ]


def test_launch_existing_sweep_dispatches_to_existing_wandb_sweep(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = []

    def fake_parse_args():
        class Args:
            config_path = "configs/sweeps/toy_sweep_001"

        return Args()

    def fake_resolve_config(path: Path) -> Path:
        calls.append(("resolve", path))
        return Path("configs/sweeps/toy_sweep_001.yaml")

    def fake_load_config(path: Path) -> dict:
        calls.append(("load", path))
        return {"sweep_id": "abc123"}

    def fake_run_existing_wandb_sweep(config: dict) -> None:
        calls.append(("existing_sweep", config))

    monkeypatch.setattr(launch, "_parse_args", fake_parse_args)
    monkeypatch.setattr(launch, "resolve_config", fake_resolve_config)
    monkeypatch.setattr(launch, "load_config", fake_load_config)
    monkeypatch.setattr(
        launch,
        "run_existing_wandb_sweep",
        fake_run_existing_wandb_sweep,
    )

    launch.main()

    assert calls == [
        ("resolve", Path("configs/sweeps/toy_sweep_001").expanduser().resolve()),
        ("load", Path("configs/sweeps/toy_sweep_001.yaml")),
        ("existing_sweep", {"sweep_id": "abc123"}),
    ]