from pathlib import Path

from Octave.src.Setup.config_resolution import load_config


RUN_CONFIG_DIR = Path(__file__).resolve().parents[2] / "configs" / "runs"


def load_run_config(name: str) -> dict:
    return load_config(RUN_CONFIG_DIR / name)


def test_ac_video_jepa_train_config_has_expected_runtime_sections() -> None:
    config = load_run_config("ac_video_jepa_train.yaml")

    assert set(config) == {
        "datamodule",
        "module",
        "trainer",
        "loggers",
        "checkpoints",
        "resume",
        "loading",
        "setup",
    }
    assert "sweep" not in config


def test_ac_video_jepa_train_config_enables_wandb_and_checkpoints() -> None:
    config = load_run_config("ac_video_jepa_train.yaml")

    assert config["setup"]["logger_registration"]["wandb"]["enabled"] is True
    assert config["loggers"]["wandb"]["offline"] is False
    assert config["loggers"]["wandb"]["watch"]["enabled"] is True
    assert config["loggers"]["wandb"]["watch"]["log"] == "all"
    assert set(config["checkpoints"]) == {
        "last",
        "periodic",
        "best_val_loss",
    }
    assert config["resume"]["enabled"] is False
    assert config["loading"]["module"]["enabled"] is False


def test_ac_video_jepa_debug_config_is_short_non_fast_dev_run() -> None:
    config = load_run_config("ac_video_jepa_debug.yaml")

    assert "fast_dev_run" not in config["trainer"]
    assert config["trainer"]["max_steps"] == 2
    assert config["trainer"]["limit_val_batches"] == 1
    assert config["loggers"]["csv"]["flush_logs_every_n_steps"] == 1
    assert set(config["checkpoints"]) == {"last"}
    assert config["setup"]["logger_registration"]["wandb"]["enabled"] is False


def test_run_configs_use_safe_cuda_dataloader_settings() -> None:
    run_configs = [
        load_run_config("ac_video_jepa_train.yaml"),
        load_run_config("ac_video_jepa_validate.yaml"),
    ]

    for config in run_configs:
        for phase, dataset_config in config["datamodule"]["datasets"].items():
            if dataset_config is None:
                continue

            dataloader_config = config["datamodule"]["dataloader_configs"][phase]

            if dataset_config["device"] == "cuda":
                assert dataloader_config["num_workers"] == 0
                assert dataloader_config["persistent_workers"] is False


def test_ac_video_jepa_train_config_matches_ac_model_contract() -> None:
    config = load_run_config("ac_video_jepa_train.yaml")
    model_config = config["module"]["model_config"]

    assert model_config["model_type"] == "ac_video_jepa"
    assert model_config["encoder"]["encoder_type"] == "impala"
    assert model_config["predictor"]["predictor_type"] == "rnn"
    assert config["module"]["unroll_config"]["nsteps"] == 8


def test_ac_video_jepa_validate_config_loads_module_without_resume() -> None:
    config = load_run_config("ac_video_jepa_validate.yaml")

    assert "sweep" not in config
    assert config["resume"]["enabled"] is False
    assert config["loading"]["module"]["enabled"] is True
    assert config["loading"]["module"]["type"] == "lightning_module"
    assert config["loading"]["module"]["relative_to"] == "run_root"
    assert config["checkpoints"] == {}


def test_ac_video_jepa_validate_config_uses_validation_phase_only() -> None:
    config = load_run_config("ac_video_jepa_validate.yaml")

    assert config["datamodule"]["datasets"]["train"] is None
    assert config["datamodule"]["datasets"]["val"]["dataset_type"] == "two_rooms"
    assert config["datamodule"]["collators"]["val"]["collator_type"] == "ac_video_jepa"
