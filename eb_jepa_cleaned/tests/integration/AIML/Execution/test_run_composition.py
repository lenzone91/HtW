"""
Integration test for AIML.Execution.Runs.

The Phase 2 capstone: compose a full run from a resolved plain-dict config with
generic dummy objects, and run a real 1-step Lightning fit:

    config -> build_training_objects
      -> datamodule (dataset -> collator)
      -> module (model built via field resolution)
      -> Trainer
    -> trainer.fit (one step)

The full end-to-end smoke test with real audio configs is Phase 4.
"""

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from src.AIML.Data.Collators.base import (
    BaseCollator,
)
from src.AIML.Data.Collators.registry import (
    COLLATOR_REGISTRY,
)
from src.AIML.Data.Datasets.base import (
    BaseDataset,
)
from src.AIML.Data.Datasets.registry import (
    DATASET_REGISTRY,
)
from src.AIML.Execution.Runs.factory import (
    build_training_objects,
)
from src.AIML.Models.Models.factory import (
    build_model,
)
from src.AIML.Models.Models.registry import (
    MODEL_REGISTRY,
)
from src.AIML.Models.Modules.base import (
    BaseLightningModule,
)
from src.AIML.Models.Modules.registry import (
    LIGHTNING_MODULE_REGISTRY,
)
from src.Workflow.Factory.registry import (
    FieldResolution,
)


FEATURES = 4


#############################################
# Dummy generic objects
#############################################


class RegressionDataset(BaseDataset):
    def __init__(self, length=4):
        super().__init__()
        self.length = length

    def __len__(self):
        return self.length

    def __getitem__(self, i):
        return self.build_sample(
            input=torch.randn(FEATURES), target=torch.randn(FEATURES)
        )


class StackCollator(BaseCollator):
    def collate_samples(self, samples):
        return {
            "input": torch.stack([s["input"] for s in samples]),
            "target": torch.stack([s["target"] for s in samples]),
        }


class Linear(nn.Module):
    def __init__(self, in_dim=FEATURES, out_dim=FEATURES):
        super().__init__()
        self.lin = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        return self.lin(x)


class RegressionModule(BaseLightningModule):
    def __init__(self, model, optimizer_configs, scheduler_configs):
        super().__init__(
            models={"model": model},
            optimizer_configs=optimizer_configs,
            scheduler_configs=scheduler_configs,
        )

    def forward(self, x):
        return self.models["model"](x)

    def training_step(self, batch, batch_idx):
        preds = self(batch["input"])
        return F.mse_loss(preds, batch["target"])


def _resolve_model(config, runtime_context=None, **kwargs):
    return build_model(
        config["model_config"], model_name="linear", runtime_context=runtime_context
    )


@pytest.fixture
def registered():
    DATASET_REGISTRY.add_entry(
        name="regression_ds",
        object_cls=RegressionDataset,
        default_config={"length": None},
    )
    COLLATOR_REGISTRY.add_entry(
        name="stack", object_cls=StackCollator, default_config={}
    )
    MODEL_REGISTRY.add_entry(
        name="linear",
        object_cls=Linear,
        default_config={"in_dim": FEATURES, "out_dim": FEATURES},
    )
    LIGHTNING_MODULE_REGISTRY.add_entry(
        name="regression",
        object_cls=RegressionModule,
        default_config={
            "module_type": None,
            "model_config": None,
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
        ),
    )
    yield
    DATASET_REGISTRY.entries.pop("regression_ds", None)
    COLLATOR_REGISTRY.entries.pop("stack", None)
    MODEL_REGISTRY.entries.pop("linear", None)
    LIGHTNING_MODULE_REGISTRY.entries.pop("regression", None)


def _config():
    return {
        "datamodule": {
            "default": {
                "datasets": {
                    "train": {"regression_ds": {"length": 4}},
                    "val": None,
                    "test": None,
                },
                "collators": {
                    "train": {"stack": {}},
                    "val": None,
                    "test": None,
                },
                "dataloader_configs": {
                    "train": {"batch_size": 2},
                    "val": {},
                    "test": {},
                },
            }
        },
        "module": {
            "regression": {
                "module_type": "regression",
                "model_config": {"in_dim": FEATURES, "out_dim": FEATURES},
                "optimizer_configs": {"model": {"optimizer_type": "adam", "lr": 0.01}},
                "scheduler_configs": {},
            }
        },
        "trainer": {
            "max_epochs": 1,
            "limit_train_batches": 1,
            # No val set in this dummy run -> disable validation.
            "limit_val_batches": 0,
            "num_sanity_val_steps": 0,
            "accelerator": "cpu",
            "devices": 1,
            "enable_checkpointing": False,
            "enable_progress_bar": False,
            "enable_model_summary": False,
        },
    }


#############################################
# Tests
#############################################


def test_build_training_objects_types(registered):
    from lightning import Trainer

    from src.AIML.Data.DataModules.base import (
        BaseDataModule,
    )

    objects = build_training_objects(_config())

    assert isinstance(objects["trainer"], Trainer)
    assert isinstance(objects["module"], RegressionModule)
    assert isinstance(objects["datamodule"], BaseDataModule)


def test_one_step_fit_runs(registered):
    objects = build_training_objects(_config())

    objects["trainer"].fit(
        model=objects["module"], datamodule=objects["datamodule"]
    )

    assert objects["trainer"].global_step >= 1
