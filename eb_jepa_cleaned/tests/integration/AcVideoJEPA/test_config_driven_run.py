"""
Config-driven AcVideoJEPA run: the full intended flow.

    Hydra compose (Workflow/Configs) -> resolved plain dict
      -> Workflow/Setup.build_runtime_context
      -> AIML Execution.build_training_objects
      -> trainer.fit on real (tiny) two-rooms data.
"""

from pathlib import Path

# Importing the experiment package registers all its concretes onto AIML.
import src.AcVideoJEPA as acvideo_jepa
from src.AIML.Execution.Runs.factory import build_training_objects
from src.Workflow.Configs.compose import load_resolved_config
from src.Workflow.Setup import build_runtime_context

CONFIG_DIR = Path(acvideo_jepa.__file__).parent / "configs"


def test_hydra_compose_yields_expected_sections():
    config = load_resolved_config(CONFIG_DIR, "config")
    assert set(config) >= {"setup", "datamodule", "module", "trainer"}
    assert "ac_video_jepa" in config["module"]
    assert "default" in config["datamodule"]
    assert config["setup"]["reproducibility"]["seed"] == 42


def test_hydra_overrides_apply():
    config = load_resolved_config(
        CONFIG_DIR, "config", overrides=["trainer.max_steps=3", "setup.reproducibility.seed=7"]
    )
    assert config["trainer"]["max_steps"] == 3
    assert config["setup"]["reproducibility"]["seed"] == 7


def test_config_driven_run_one_step(tmp_path):
    config = load_resolved_config(CONFIG_DIR, "config")
    # Point the run directory at a temp location (a launcher would override the
    # existing-run-dir policy similarly).
    config["setup"]["paths"]["project_root"] = str(tmp_path)
    config["setup"]["paths"]["existing_run_dir_policy"] = "overwrite"

    runtime_context = build_runtime_context(config["setup"])
    assert Path(runtime_context["paths"]["run_dir"]).is_dir()

    objects = build_training_objects(config, runtime_context=runtime_context)
    objects["trainer"].fit(objects["module"], objects["datamodule"])
    assert objects["trainer"].state.finished
