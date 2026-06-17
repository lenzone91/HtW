"""
Integration test for Workflow.Factory.

Exercises the full intended flow: a plain Python config dict (the shape produced
downstream of the Hydra -> dict boundary) is turned into a built object graph
through linked registries and builders, including type-field routing, field
resolution, and nested sub-builds.
"""

import pytest

from eb_jepa_cleaned.Workflow.Factory.builder import (
    RegistryBuilder,
)
from eb_jepa_cleaned.Workflow.Factory.errors import (
    FactoryError,
)
from eb_jepa_cleaned.Workflow.Factory.registry import (
    FieldResolution,
    Registry,
    SubBuildDeclaration,
)


#############################################
# A small object graph: Model -> Encoder
#############################################


class LinearEncoder:
    def __init__(self, in_dim, out_dim, dtype="float32"):
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.dtype = dtype


class ConvEncoder:
    def __init__(self, channels, dtype="float32"):
        self.channels = channels
        self.dtype = dtype


class Model:
    def __init__(self, encoder, n_layers):
        self.encoder = encoder
        self.n_layers = n_layers


# Resolver mapping serializable dtype names to a normalized value, demonstrating
# field resolution at the boundary (no DictConfig, no runtime objects in config).
DTYPE_ALIASES = {"f32": "float32", "f64": "float64"}


def _resolve_dtype(config, runtime_context, **kwargs):
    return DTYPE_ALIASES[config["dtype"]]


@pytest.fixture
def model_builder():
    encoder_registry = Registry("encoder")
    encoder_registry.add_entry(
        name="linear",
        object_cls=LinearEncoder,
        default_config={"type": None, "in_dim": None, "out_dim": None, "dtype": "f32"},
        type_field="type",
        field_resolutions=(
            FieldResolution(target_key="dtype", resolver=_resolve_dtype),
        ),
    )
    encoder_registry.add_entry(
        name="conv",
        object_cls=ConvEncoder,
        default_config={"type": None, "channels": None, "dtype": "f32"},
        type_field="type",
        field_resolutions=(
            FieldResolution(target_key="dtype", resolver=_resolve_dtype),
        ),
    )
    encoder_builder = RegistryBuilder(encoder_registry, type_field="type")

    model_registry = Registry("model")
    model_registry.add_entry(
        name="model",
        object_cls=Model,
        default_config={"encoder": None, "n_layers": 1},
        sub_builds=(
            SubBuildDeclaration(
                source_key="encoder",
                target_key="encoder",
                builder=encoder_builder,
                build_method="one",
            ),
        ),
    )
    return RegistryBuilder(model_registry)


def test_config_dict_builds_full_object_graph(model_builder):
    config = {
        "encoder": {"type": "linear", "in_dim": 16, "out_dim": 8, "dtype": "f64"},
        "n_layers": 3,
    }

    model = model_builder.build_one(config, name="model")

    assert isinstance(model, Model)
    assert model.n_layers == 3
    assert isinstance(model.encoder, LinearEncoder)
    assert (model.encoder.in_dim, model.encoder.out_dim) == (16, 8)
    # Field resolution applied and routing key stripped before construction.
    assert model.encoder.dtype == "float64"


def test_encoder_type_routing_selects_variant(model_builder):
    config = {
        "encoder": {"type": "conv", "channels": 32, "dtype": "f32"},
        "n_layers": 1,
    }

    model = model_builder.build_one(config, name="model")

    assert isinstance(model.encoder, ConvEncoder)
    assert model.encoder.channels == 32
    assert model.encoder.dtype == "float32"


def test_invalid_nested_key_fails_strictly(model_builder):
    config = {
        "encoder": {"type": "linear", "in_dim": 1, "out_dim": 1, "dtype": "f32", "oops": 1},
        "n_layers": 1,
    }

    with pytest.raises(FactoryError, match="Invalid config keys"):
        model_builder.build_one(config, name="model")


def test_original_config_not_mutated(model_builder):
    config = {
        "encoder": {"type": "linear", "in_dim": 16, "out_dim": 8, "dtype": "f64"},
        "n_layers": 3,
    }
    snapshot = {
        "encoder": {"type": "linear", "in_dim": 16, "out_dim": 8, "dtype": "f64"},
        "n_layers": 3,
    }

    model_builder.build_one(config, name="model")

    assert config == snapshot
