from pathlib import Path

import yaml

from Octave import launch


DEBUG_RUN_CONFIG_PATH = (
    Path(__file__).resolve().parents[2]
    / "configs"
    / "runs"
    / "ac_video_jepa_debug.yaml"
)


def test_launch_debug_run_writes_report_checkpoint_and_metrics(
    tmp_path: Path,
    monkeypatch,
) -> None:
    config = load_debug_run_config()
    config["setup"]["paths"]["project_root"] = str(tmp_path)
    config["setup"]["paths"]["run_root"] = "runs"
    config["setup"]["paths"]["run_name"] = "debug_launch"

    config_path = tmp_path / "debug.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

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

    launch.main()

    run_dir = tmp_path / "runs" / "ac_video_jepa" / "debug_launch"
    report_path = run_dir / "train_execution_report.json"
    checkpoint_path = run_dir / "checkpoints" / "last.ckpt"
    metric_paths = list((run_dir / "logs").rglob("metrics.csv"))

    assert report_path.exists()
    assert checkpoint_path.exists()
    assert metric_paths
    assert any("train/loss" in path.read_text(encoding="utf-8") for path in metric_paths)


def load_debug_run_config() -> dict:
    with DEBUG_RUN_CONFIG_PATH.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


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
