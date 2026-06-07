"""
Pytest test module for Configs.resolve.

The tests are intentionally written without parametrization to keep each
resolution behavior explicit and easy to debug.
"""

from pathlib import Path

import pytest

from Project_1_SepFormer_TSE_HF.idea_1_project_setup.src.Configs.conversion import (
    load_config,
    save_config,
)
from Project_1_SepFormer_TSE_HF.idea_1_project_setup.src.Configs.resolve import (
    resolve_config,
)


#############################################
# Helpers
#############################################

def _write_config(path: Path, config: dict) -> None:
    """
    Write a YAML config fixture.
    """
    save_config(config=config, path=path)


#############################################
# File input tests
#############################################

def test_resolve_file_returns_same_path(tmp_path: Path) -> None:
    """
    Test that resolving an existing config file returns the same path.
    """
    config_path = tmp_path / "config.yaml"
    _write_config(config_path, {"trainer": {"max_epochs": 1}})

    resolved_path = resolve_config(config_path)

    assert resolved_path == config_path.resolve()


#############################################
# Folder resolution tests
#############################################

def test_resolve_folder_writes_sibling_yaml(tmp_path: Path) -> None:
    """
    Test that resolving a folder writes a sibling global YAML config.
    """
    config_dir = tmp_path / "toy_run_001"
    config_dir.mkdir()

    _write_config(config_dir / "execution.yaml", {"trainer": {"max_epochs": 1}})

    resolved_path = resolve_config(config_dir)

    assert resolved_path == (tmp_path / "toy_run_001.yaml").resolve()
    assert resolved_path.is_file()


def test_resolve_folder_resolves_top_level_references(tmp_path: Path) -> None:
    """
    Test that top-level same-folder YAML references are resolved.
    """
    config_dir = tmp_path / "toy_run_001"
    config_dir.mkdir()

    _write_config(
        config_dir / "execution.yaml",
        {
            "setup": "setup.yaml",
            "trainer": "trainer.yaml",
        },
    )
    _write_config(config_dir / "setup.yaml", {"paths": {"run_name": "toy_run_001"}})
    _write_config(config_dir / "trainer.yaml", {"max_epochs": 1})

    resolved_path = resolve_config(config_dir)
    resolved_config = load_config(resolved_path)

    assert resolved_config == {
        "setup": {"paths": {"run_name": "toy_run_001"}},
        "trainer": {"max_epochs": 1},
    }


def test_resolve_folder_resolves_nested_references(tmp_path: Path) -> None:
    """
    Test that nested same-folder YAML references are resolved.
    """
    config_dir = tmp_path / "toy_run_001"
    config_dir.mkdir()

    _write_config(config_dir / "execution.yaml", {"module": "module.yaml"})
    _write_config(
        config_dir / "module.yaml",
        {
            "waveunet": {
                "model_configs": {
                    "model": "model.yaml",
                },
                "train_metrics_config": "train_metrics.yaml",
            }
        },
    )
    _write_config(
        config_dir / "model.yaml",
        {
            "waveunet": {
                "Fc": 4,
                "fd": 5,
                "fu": 5,
                "L": 2,
            }
        },
    )
    _write_config(
        config_dir / "train_metrics.yaml",
        {
            "set_type": "tse",
            "strict": True,
            "metrics": {
                "snr": {
                    "zero_mean": False,
                    "reduction": "mean",
                }
            },
        },
    )

    resolved_path = resolve_config(config_dir)
    resolved_config = load_config(resolved_path)

    assert resolved_config == {
        "module": {
            "waveunet": {
                "model_configs": {
                    "model": {
                        "waveunet": {
                            "Fc": 4,
                            "fd": 5,
                            "fu": 5,
                            "L": 2,
                        }
                    }
                },
                "train_metrics_config": {
                    "set_type": "tse",
                    "strict": True,
                    "metrics": {
                        "snr": {
                            "zero_mean": False,
                            "reduction": "mean",
                        }
                    },
                },
            }
        }
    }


def test_resolve_folder_resolves_recursive_references(tmp_path: Path) -> None:
    """
    Test that references inside referenced files are also resolved.
    """
    config_dir = tmp_path / "toy_run_001"
    config_dir.mkdir()

    _write_config(config_dir / "execution.yaml", {"module": "module.yaml"})
    _write_config(config_dir / "module.yaml", {"loss_weights": "loss.yaml"})
    _write_config(config_dir / "loss.yaml", {"metric_weights": "metric_weights.yaml"})
    _write_config(config_dir / "metric_weights.yaml", {"snr": -1.0, "lp_error": 1.0})

    resolved_path = resolve_config(config_dir)
    resolved_config = load_config(resolved_path)

    assert resolved_config == {
        "module": {
            "loss_weights": {
                "metric_weights": {
                    "snr": -1.0,
                    "lp_error": 1.0,
                }
            }
        }
    }


def test_resolve_folder_preserves_non_reference_strings(tmp_path: Path) -> None:
    """
    Test that ordinary strings are not treated as config references.
    """
    config_dir = tmp_path / "toy_run_001"
    config_dir.mkdir()

    _write_config(
        config_dir / "execution.yaml",
        {
            "setup": {
                "paths": {
                    "experiment_name": "toy_experiment",
                    "run_name": "toy_run_001",
                }
            },
            "logger": {
                "project": "tse-toy",
            },
        },
    )

    resolved_path = resolve_config(config_dir)
    resolved_config = load_config(resolved_path)

    assert resolved_config == {
        "setup": {
            "paths": {
                "experiment_name": "toy_experiment",
                "run_name": "toy_run_001",
            }
        },
        "logger": {
            "project": "tse-toy",
        },
    }


def test_resolve_preserves_path_like_yaml_strings(tmp_path: Path) -> None:
    """
    Test that YAML-looking strings under path-like keys are not resolved.

    This prevents private credential files from being inlined into the
    resolved experiment config.
    """
    config_dir = tmp_path / "toy_run_001"
    config_dir.mkdir()

    _write_config(
        config_dir / "execution.yaml",
        {
            "setup": {
                "user_credential": {
                    "path": "user_credential.yaml",
                }
            }
        },
    )
    _write_config(
        config_dir / "user_credential.yaml",
        {
            "wandb": {
                "api_key": "secret",
            }
        },
    )

    resolved_path = resolve_config(config_dir)
    resolved_config = load_config(resolved_path)

    assert resolved_config == {
        "setup": {
            "user_credential": {
                "path": "user_credential.yaml",
            }
        }
    }


def test_resolve_still_resolves_non_path_yaml_reference(tmp_path: Path) -> None:
    """
    Test that regular config references are still resolved after path exclusions.
    """
    config_dir = tmp_path / "toy_run_001"
    config_dir.mkdir()

    _write_config(config_dir / "execution.yaml", {"setup": "setup.yaml"})
    _write_config(config_dir / "setup.yaml", {"paths": {"run_name": "toy_run_001"}})

    resolved_path = resolve_config(config_dir)
    resolved_config = load_config(resolved_path)

    assert resolved_config == {
        "setup": {
            "paths": {
                "run_name": "toy_run_001",
            }
        }
    }


#############################################
# Error tests
#############################################

def test_resolve_missing_input_path_raises(tmp_path: Path) -> None:
    """
    Test that resolving a missing path raises an error.
    """
    with pytest.raises(FileNotFoundError):
        resolve_config(tmp_path / "missing.yaml")


def test_resolve_folder_without_execution_yaml_raises(tmp_path: Path) -> None:
    """
    Test that a config folder must contain execution.yaml.
    """
    config_dir = tmp_path / "toy_run_001"
    config_dir.mkdir()

    with pytest.raises(FileNotFoundError):
        resolve_config(config_dir)


def test_resolve_missing_referenced_file_raises(tmp_path: Path) -> None:
    """
    Test that missing referenced config files raise an error.
    """
    config_dir = tmp_path / "toy_run_001"
    config_dir.mkdir()

    _write_config(config_dir / "execution.yaml", {"setup": "setup.yaml"})

    with pytest.raises(FileNotFoundError):
        resolve_config(config_dir)


def test_resolve_reference_cycle_raises(tmp_path: Path) -> None:
    """
    Test that cyclic config references raise an error.
    """
    config_dir = tmp_path / "toy_run_001"
    config_dir.mkdir()

    _write_config(config_dir / "execution.yaml", {"a": "a.yaml"})
    _write_config(config_dir / "a.yaml", {"b": "b.yaml"})
    _write_config(config_dir / "b.yaml", {"a": "a.yaml"})

    with pytest.raises(ValueError):
        resolve_config(config_dir)


def test_resolve_unsupported_input_extension_raises(tmp_path: Path) -> None:
    """
    Test that unsupported direct config file extensions raise an error.
    """
    config_path = tmp_path / "config.txt"
    config_path.write_text("not a supported config", encoding="utf-8")

    with pytest.raises(ValueError):
        resolve_config(config_path)


def test_resolve_unsupported_reference_extension_is_not_resolved(tmp_path: Path) -> None:
    """
    Test that strings with unsupported extensions are preserved as ordinary strings.
    """
    config_dir = tmp_path / "toy_run_001"
    config_dir.mkdir()

    _write_config(config_dir / "execution.yaml", {"notes": "notes.txt"})

    resolved_path = resolve_config(config_dir)
    resolved_config = load_config(resolved_path)

    assert resolved_config == {"notes": "notes.txt"}