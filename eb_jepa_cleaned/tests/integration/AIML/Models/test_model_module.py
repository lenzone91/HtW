"""
Integration test for AIML model + module wiring.

A concrete (dummy) Lightning module is built through the factory; its model,
loss, and metric set are sub-built as already-built objects via field
resolutions (mirroring the audio waveunet module), while optimizer/scheduler
configs are kept for configure_optimizers. Exercises Decision 15.
"""

import pytest
import torch
from torch import nn

from eb_jepa_cleaned.AIML.Metrics.Loss.factory import (
    build_loss,
)
from eb_jepa_cleaned.AIML.Metrics.Loss.loss import (
    WeightedMetricLoss,
)
from eb_jepa_cleaned.AIML.Metrics.Metrics.base import (
    BaseMetric,
)
from eb_jepa_cleaned.AIML.Metrics.Metrics.registry import (
    METRIC_REGISTRY,
)
from eb_jepa_cleaned.AIML.Metrics.MetricSets.factory import (
    build_metric_set,
)
from eb_jepa_cleaned.AIML.Metrics.MetricSets.metric_set import (
    MetricSet,
)
from eb_jepa_cleaned.AIML.Models.Models.factory import (
    build_model,
)
from eb_jepa_cleaned.AIML.Models.Models.registry import (
    MODEL_REGISTRY,
)
from eb_jepa_cleaned.AIML.Models.Modules.base import (
    BaseLightningModule,
)
from eb_jepa_cleaned.AIML.Models.Modules.factory import (
    build_lightning_module,
)
from eb_jepa_cleaned.AIML.Models.Modules.registry import (
    LIGHTNING_MODULE_REGISTRY,
)
from eb_jepa_cleaned.Workflow.Factory.registry import (
    FieldResolution,
)


#############################################
# Dummy generic objects
#############################################


class Tiny(nn.Module):
    def __init__(self, in_dim=2, out_dim=2):
        super().__init__()
        self.lin = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        return self.lin(x)


class ConstMetric(BaseMetric):
    def __init__(self, reduction="mean", value=1.0):
        super().__init__(reduction=reduction)
        self.value = value

    def forward(self, preds, target):
        return torch.tensor(self.value)


class DummyModule(BaseLightningModule):
    def __init__(self, model, loss, metric_set, optimizer_configs, scheduler_configs):
        super().__init__(
            models={"model": model},
            optimizer_configs=optimizer_configs,
            scheduler_configs=scheduler_configs,
        )
        self.loss = loss
        self.metric_set = metric_set

    def forward(self, x):
        return self.models["model"](x)


#############################################
# Field-resolution wiring (built sub-objects)
#############################################


def _resolve_model(config, runtime_context=None, **kwargs):
    return build_model(
        config["model_config"], model_name="tiny", runtime_context=runtime_context
    )


def _resolve_loss(config, runtime_context=None, **kwargs):
    return build_loss(config["loss_config"], runtime_context=runtime_context)


def _resolve_metric_set(config, runtime_context=None, **kwargs):
    return build_metric_set(config["metrics_config"], runtime_context=runtime_context)


@pytest.fixture
def registered():
    MODEL_REGISTRY.add_entry(
        name="tiny", object_cls=Tiny, default_config={"in_dim": 2, "out_dim": 2}
    )
    METRIC_REGISTRY.add_entry(
        name="const",
        object_cls=ConstMetric,
        default_config={"reduction": "mean", "value": 1.0},
    )
    LIGHTNING_MODULE_REGISTRY.add_entry(
        name="dummy",
        object_cls=DummyModule,
        default_config={
            "module_type": None,
            "model_config": None,
            "loss_config": None,
            "metrics_config": None,
            "optimizer_configs": None,
            "scheduler_configs": None,
        },
        type_field="module_type",
        field_resolutions=(
            FieldResolution(
                target_key="model",
                resolver=_resolve_model,
                remove_source_keys=("model_config",),
            ),
            FieldResolution(
                target_key="loss",
                resolver=_resolve_loss,
                remove_source_keys=("loss_config",),
            ),
            FieldResolution(
                target_key="metric_set",
                resolver=_resolve_metric_set,
                remove_source_keys=("metrics_config",),
            ),
        ),
    )
    yield
    MODEL_REGISTRY.entries.pop("tiny", None)
    METRIC_REGISTRY.entries.pop("const", None)
    LIGHTNING_MODULE_REGISTRY.entries.pop("dummy", None)


def _config():
    return {
        "dummy": {
            "module_type": "dummy",
            "model_config": {"in_dim": 2, "out_dim": 2},
            "loss_config": {
                "loss_type": "weighted_metric",
                "metric_weights": {"const": 1.0},
            },
            "metrics_config": {"set_type": "metric", "metrics": {"const": {}}},
            "optimizer_configs": {"model": {"optimizer_type": "adam", "lr": 0.01}},
            "scheduler_configs": {},
        }
    }


#############################################
# Tests
#############################################


def test_module_receives_built_sub_objects(registered):
    module = build_lightning_module(_config())

    assert isinstance(module, DummyModule)
    assert isinstance(module.models["model"], Tiny)
    assert isinstance(module.loss, WeightedMetricLoss)
    assert isinstance(module.metric_set, MetricSet)


def test_built_module_forward_and_optimizers(registered):
    module = build_lightning_module(_config())

    output = module(torch.zeros(3, 2))
    assert output.shape == (3, 2)

    optimizers = module.configure_optimizers()
    assert isinstance(optimizers[0], torch.optim.Adam)


def test_built_metric_set_evaluates(registered):
    module = build_lightning_module(_config())

    values = module.metric_set({"const": (torch.zeros(2), torch.zeros(2))})
    assert values["const"] == 1.0
