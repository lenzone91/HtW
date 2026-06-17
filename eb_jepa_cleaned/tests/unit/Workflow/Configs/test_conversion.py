"""
Unit tests for Workflow.Configs.conversion.

Covers plain-dict file load/save round-trips and the strict error surface.
"""

import pytest

from eb_jepa_cleaned.Workflow.Configs.conversion import (
    load_config,
    save_config,
)
from eb_jepa_cleaned.Workflow.Configs.errors import (
    ConfigError,
)


@pytest.mark.parametrize("suffix", [".yaml", ".yml", ".json"])
def test_save_load_round_trip(tmp_path, suffix):
    config = {"a": 1, "nested": {"b": [1, 2, 3], "c": "x"}}
    path = tmp_path / f"config{suffix}"

    save_config(config, path)
    loaded = load_config(path)

    assert loaded == config


def test_load_returns_plain_dict(tmp_path):
    path = tmp_path / "c.yaml"
    save_config({"a": {"b": 1}}, path)

    loaded = load_config(path)

    assert isinstance(loaded, dict)
    assert isinstance(loaded["a"], dict)


def test_load_unsupported_extension_raises(tmp_path):
    path = tmp_path / "c.txt"
    path.write_text("a: 1", encoding="utf-8")

    with pytest.raises(ConfigError, match="Unsupported config file extension"):
        load_config(path)


def test_save_unsupported_extension_raises(tmp_path):
    with pytest.raises(ConfigError, match="Unsupported config file extension"):
        save_config({"a": 1}, tmp_path / "c.txt")


def test_load_non_mapping_raises(tmp_path):
    path = tmp_path / "list.yaml"
    path.write_text("- 1\n- 2\n", encoding="utf-8")

    with pytest.raises(ConfigError, match="must be a dictionary"):
        load_config(path)


def test_save_non_dict_raises(tmp_path):
    with pytest.raises(ConfigError, match="must be a dictionary"):
        save_config([1, 2, 3], tmp_path / "c.yaml")


def test_save_creates_parent_dirs(tmp_path):
    path = tmp_path / "deep" / "nested" / "c.yaml"

    save_config({"a": 1}, path)

    assert path.is_file()
