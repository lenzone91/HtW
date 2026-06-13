from copy import deepcopy
from pathlib import Path

import yaml

from Octave.src.Execution.train import run_training
from Octave.src.Execution.validate import run_validation


TOY_RUN_CONFIG_PATH = (
    Path(__file__).resolve().parents[2]
    / "configs"
    / "runs"
    / "ac_video_jepa_toy.yaml"
)


def load_toy_run_config() -> dict:
    with TOY_RUN_CONFIG_PATH.open("r") as file:
        return yaml.safe_load(file)


def test_ac_video_jepa_toy_config_fast_dev_run_fit(tmp_path: Path) -> None:
    config = load_toy_run_config()
    config["setup"]["paths"]["project_root"] = str(tmp_path)
    config["setup"]["paths"]["run_root"] = "runs"
    config["setup"]["paths"]["run_name"] = "train_run"

    report = run_training(config=config)

    assert report["status"] == "finished"


def test_ac_video_jepa_toy_config_fast_dev_run_validate(tmp_path: Path) -> None:
    config = load_toy_run_config()
    config["setup"]["paths"]["project_root"] = str(tmp_path)
    config["setup"]["paths"]["run_root"] = "runs"
    config["setup"]["paths"]["run_name"] = "validate_run"

    report = run_validation(config=config)

    assert report["status"] == "finished"
    assert isinstance(report["outputs"], list)


def test_ac_video_jepa_toy_config_path_fast_dev_run_fit(tmp_path: Path) -> None:
    config = load_toy_run_config()
    config["setup"]["paths"]["project_root"] = str(tmp_path)
    config["setup"]["paths"]["run_root"] = "runs"
    config["setup"]["paths"]["run_name"] = "config_path_train_run"

    config_path = tmp_path / "toy.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    report = run_training(config_path=config_path)

    assert report["status"] == "finished"


def test_ac_video_jepa_toy_checkpoint_can_be_loaded_for_validation(
    tmp_path: Path,
) -> None:
    train_config = load_toy_run_config()
    train_config["setup"]["paths"]["project_root"] = str(tmp_path)
    train_config["setup"]["paths"]["run_root"] = "runs"
    train_config["setup"]["paths"]["run_name"] = "train_with_checkpoint"
    train_config["trainer"].pop("fast_dev_run")
    train_config["trainer"].update(
        {
            "max_steps": 1,
            "limit_val_batches": 1,
            "num_sanity_val_steps": 0,
        }
    )

    train_report = run_training(config=train_config)
    checkpoint_path = (
        tmp_path
        / "runs"
        / "ac_video_jepa"
        / "train_with_checkpoint"
        / "checkpoints"
        / "last.ckpt"
    )

    assert train_report["status"] == "finished"
    assert checkpoint_path.exists()

    validate_config = deepcopy(train_config)
    validate_config["setup"]["paths"]["run_name"] = "validate_loaded_checkpoint"
    validate_config["trainer"].pop("max_steps")
    validate_config["trainer"].pop("limit_val_batches")
    validate_config["trainer"]["enable_checkpointing"] = False
    validate_config["datamodule"]["datasets"]["train"] = None
    validate_config["datamodule"]["collators"]["train"] = None
    validate_config["datamodule"]["dataloader_configs"]["train"] = None
    validate_config["checkpoints"] = {}
    validate_config["loading"]["module"].update(
        {
            "enabled": True,
            "checkpoint_path": "ac_video_jepa/train_with_checkpoint/checkpoints/last.ckpt",
            "relative_to": "run_root",
        }
    )

    validate_report = run_validation(config=validate_config)

    assert validate_report["status"] == "finished"
    assert isinstance(validate_report["outputs"], list)
