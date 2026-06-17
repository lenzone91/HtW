"""Workflow/Setup: device, reproducibility, paths, and runtime_context."""

import random
from pathlib import Path

import numpy as np
import pytest
import torch

from src.Workflow.Setup import build_runtime_context
from src.Workflow.Setup.device import resolve_device
from src.Workflow.Setup.paths import setup_paths
from src.Workflow.Setup.reproducibility import setup_reproducibility


#############################################
# Device
#############################################


def test_resolve_device_auto_is_cpu_without_cuda():
    expected = "cuda" if torch.cuda.is_available() else "cpu"
    assert resolve_device("auto") == expected


def test_resolve_device_rejects_unknown():
    with pytest.raises(ValueError):
        resolve_device("tpu")


#############################################
# Reproducibility
#############################################


def test_reproducibility_seeds_and_reports():
    summary = setup_reproducibility({"seed": 123})
    assert summary == {"seed": 123, "deterministic": False, "cudnn_benchmark": False}
    # Same seed -> same draws across the seeded RNGs.
    setup_reproducibility({"seed": 7})
    draws = (random.random(), float(np.random.rand()), float(torch.rand(1)))
    setup_reproducibility({"seed": 7})
    again = (random.random(), float(np.random.rand()), float(torch.rand(1)))
    assert draws == again


#############################################
# Paths
#############################################


def test_setup_paths_creates_run_dir_and_subdirs(tmp_path):
    paths = setup_paths(
        {
            "project_root": str(tmp_path),
            "run_root": "runs",
            "experiment_name": "ac_video_jepa",
            "run_name": "r0",
            "existing_run_dir_policy": "fail",
        }
    )
    run_dir = Path(paths["run_dir"])
    assert run_dir == tmp_path / "runs" / "ac_video_jepa" / "r0"
    for key in ("run_dir", "checkpoints_dir", "logs_dir", "configs_dir", "artifacts_dir"):
        assert Path(paths[key]).is_dir()


def test_setup_paths_policy_fail_then_overwrite(tmp_path):
    base = {
        "project_root": str(tmp_path),
        "run_root": "runs",
        "run_name": "r0",
    }
    setup_paths({**base, "existing_run_dir_policy": "fail"})
    with pytest.raises(FileExistsError):
        setup_paths({**base, "existing_run_dir_policy": "fail"})
    # overwrite recreates without error
    setup_paths({**base, "existing_run_dir_policy": "overwrite"})


def test_setup_paths_rejects_unknown_policy(tmp_path):
    with pytest.raises(ValueError):
        setup_paths({"project_root": str(tmp_path), "existing_run_dir_policy": "bogus"})


#############################################
# runtime_context
#############################################


def test_build_runtime_context_assembles_sections(tmp_path):
    rc = build_runtime_context(
        {
            "device": "cpu",
            "reproducibility": {"seed": 1},
            "paths": {
                "project_root": str(tmp_path),
                "run_root": "runs",
                "run_name": "r0",
                "existing_run_dir_policy": "overwrite",
            },
        }
    )
    assert rc["device"] == "cpu"
    assert rc["reproducibility"]["seed"] == 1
    assert Path(rc["paths"]["run_dir"]).is_dir()
