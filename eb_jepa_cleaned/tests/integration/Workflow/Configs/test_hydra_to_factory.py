"""
Integration test for the Workflow config layer.

Exercises the full intended flow:

    Hydra compose (groups + defaults + overrides + interpolation)
      -> resolve to plain dict (the boundary)
      -> project factory
      -> built object

with a real on-disk Hydra config tree.
"""

import pytest

from src.Workflow.Configs.compose import (
    compose_config,
    load_resolved_config,
)
from src.Workflow.Configs.resolve import (
    check_plain_config,
    resolve_to_plain_dict,
)
from src.Workflow.Factory.builder import (
    RegistryBuilder,
)
from src.Workflow.Factory.registry import (
    Registry,
)


#############################################
# On-disk Hydra config tree fixture
#############################################


@pytest.fixture
def config_dir(tmp_path):
    (tmp_path / "model").mkdir()
    (tmp_path / "trainer").mkdir()

    (tmp_path / "config.yaml").write_text(
        "defaults:\n"
        "  - model: linear\n"
        "  - trainer: default\n"
        "  - _self_\n"
        "\n"
        "run_name: demo\n"
        "epochs: ${trainer.max_epochs}\n",
        encoding="utf-8",
    )
    (tmp_path / "model" / "linear.yaml").write_text(
        "type: linear\nin_dim: 16\nout_dim: 8\n",
        encoding="utf-8",
    )
    (tmp_path / "model" / "conv.yaml").write_text(
        "type: conv\nchannels: 32\n",
        encoding="utf-8",
    )
    (tmp_path / "trainer" / "default.yaml").write_text(
        "max_epochs: 1\n",
        encoding="utf-8",
    )
    return tmp_path


#############################################
# Composition behavior
#############################################


def test_compose_applies_defaults_and_groups(config_dir):
    resolved = load_resolved_config(config_dir, "config")

    assert resolved["run_name"] == "demo"
    assert resolved["model"]["type"] == "linear"
    assert resolved["trainer"]["max_epochs"] == 1


def test_compose_resolves_interpolation_into_plain_value(config_dir):
    resolved = load_resolved_config(config_dir, "config")

    assert resolved["epochs"] == 1
    check_plain_config(resolved)


def test_overrides_switch_group_and_set_value(config_dir):
    resolved = load_resolved_config(
        config_dir,
        "config",
        overrides=["model=conv", "trainer.max_epochs=5"],
    )

    assert resolved["model"]["type"] == "conv"
    assert resolved["model"]["channels"] == 32
    assert resolved["trainer"]["max_epochs"] == 5
    # Interpolation tracks the override.
    assert resolved["epochs"] == 5


def test_composed_config_is_dictconfig_before_boundary(config_dir):
    from omegaconf import DictConfig

    composed = compose_config(config_dir, "config")

    assert isinstance(composed, DictConfig)
    # And crossing the boundary yields a plain dict.
    assert isinstance(resolve_to_plain_dict(composed), dict)


#############################################
# Hydra config -> plain dict -> factory
#############################################


class LinearModel:
    def __init__(self, in_dim, out_dim):
        self.in_dim = in_dim
        self.out_dim = out_dim


class ConvModel:
    def __init__(self, channels):
        self.channels = channels


@pytest.fixture
def model_builder():
    registry = Registry("model")
    registry.add_entry(
        name="linear",
        object_cls=LinearModel,
        default_config={"type": None, "in_dim": None, "out_dim": None},
        type_field="type",
    )
    registry.add_entry(
        name="conv",
        object_cls=ConvModel,
        default_config={"type": None, "channels": None},
        type_field="type",
    )
    return RegistryBuilder(registry, type_field="type")


def test_resolved_config_builds_object_through_factory(config_dir, model_builder):
    resolved = load_resolved_config(config_dir, "config")

    model = model_builder.build_one(resolved["model"])

    assert isinstance(model, LinearModel)
    assert (model.in_dim, model.out_dim) == (16, 8)


def test_override_changes_built_object_type(config_dir, model_builder):
    resolved = load_resolved_config(config_dir, "config", overrides=["model=conv"])

    model = model_builder.build_one(resolved["model"])

    assert isinstance(model, ConvModel)
    assert model.channels == 32
