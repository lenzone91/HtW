from pathlib import Path

import pytest

from Octave import launch


def test_apply_existing_run_dir_policy_override_sets_overwrite() -> None:
    config = {"setup": {"paths": {"existing_run_dir_policy": "fail"}}}

    launch.apply_existing_run_dir_policy_override(
        config=config,
        overwrite=True,
    )

    assert config["setup"]["paths"]["existing_run_dir_policy"] == "overwrite"


def test_apply_existing_run_dir_policy_override_sets_ask() -> None:
    config = {}

    launch.apply_existing_run_dir_policy_override(
        config=config,
        ask_overwrite=True,
    )

    assert config["setup"]["paths"]["existing_run_dir_policy"] == "ask"


def test_launch_train_dispatches_loaded_config_with_overwrite(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = tmp_path / "run.yaml"
    config_path.write_text("{}", encoding="utf-8")
    loaded_config = {"setup": {"paths": {"existing_run_dir_policy": "fail"}}}
    calls = []

    monkeypatch.setattr(
        launch,
        "parse_args",
        lambda: make_args(
            config_path=config_path,
            mode="train",
            overwrite=True,
            ask_overwrite=False,
        ),
    )
    monkeypatch.setattr(launch, "load_config", lambda path: loaded_config)
    monkeypatch.setattr(
        launch,
        "run_training",
        lambda config, config_path: calls.append(("train", config, config_path)),
    )

    launch.main()

    assert calls == [
        (
            "train",
            {"setup": {"paths": {"existing_run_dir_policy": "overwrite"}}},
            config_path.resolve(),
        )
    ]


def test_launch_validate_dispatches_loaded_config_with_ask_overwrite(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = tmp_path / "run.yaml"
    config_path.write_text("{}", encoding="utf-8")
    loaded_config = {"setup": {"paths": {"existing_run_dir_policy": "fail"}}}
    calls = []

    monkeypatch.setattr(
        launch,
        "parse_args",
        lambda: make_args(
            config_path=config_path,
            mode="validate",
            overwrite=False,
            ask_overwrite=True,
        ),
    )
    monkeypatch.setattr(launch, "load_config", lambda path: loaded_config)
    monkeypatch.setattr(
        launch,
        "run_validation",
        lambda config, config_path: calls.append(("validate", config, config_path)),
    )

    launch.main()

    assert calls == [
        (
            "validate",
            {"setup": {"paths": {"existing_run_dir_policy": "ask"}}},
            config_path.resolve(),
        )
    ]


def test_launch_rejects_missing_config_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    missing_config_path = tmp_path / "missing.yaml"

    monkeypatch.setattr(
        launch,
        "parse_args",
        lambda: make_args(
            config_path=missing_config_path,
            mode="train",
            overwrite=False,
            ask_overwrite=False,
        ),
    )

    with pytest.raises(FileNotFoundError, match="Config file not found"):
        launch.main()


def make_args(
    config_path: Path,
    mode: str,
    overwrite: bool,
    ask_overwrite: bool,
):
    class Args:
        pass

    args = Args()
    args.config_path = config_path
    args.mode = mode
    args.overwrite = overwrite
    args.ask_overwrite = ask_overwrite
    return args
