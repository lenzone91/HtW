from pathlib import Path

import pytest

from Octave.src.Setup.config_resolution import load_config, merge_configs


def test_load_config_loads_yaml_dict(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text("a: 1\n", encoding="utf-8")

    config = load_config(config_path)

    assert config == {"a": 1}


def test_load_config_rejects_unsupported_extension(tmp_path: Path) -> None:
    config_path = tmp_path / "config.txt"
    config_path.write_text("a=1", encoding="utf-8")

    with pytest.raises(ValueError, match="Unsupported config extension"):
        load_config(config_path)


def test_merge_configs_merges_nested_dicts_without_mutating_inputs() -> None:
    base = {"a": {"b": 1}, "c": 2}
    override = {"a": {"b": 3}}

    merged = merge_configs(base, override)

    assert merged == {"a": {"b": 3}, "c": 2}
    assert base == {"a": {"b": 1}, "c": 2}
    assert override == {"a": {"b": 3}}


def test_merge_configs_rejects_unknown_keys_in_strict_mode() -> None:
    with pytest.raises(KeyError, match="Unknown config key"):
        merge_configs({"a": 1}, {"b": 2}, strict=True)
