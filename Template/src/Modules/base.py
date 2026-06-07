import torch
from torch import nn
import lightning.pytorch as pl

from ..Optimization.factory import build_optimizers_from_models, build_schedulers

######################################
# Base Lightning Module
######################################

class BaseLightningModule(pl.LightningModule):
    """
    Base PyTorch Lightning module for the TSE training pipeline.

    This class centralizes Lightning-level utilities that are shared across
    task-specific modules.

    Responsibilities:
        - define the common Lightning interface;
        - route the forward pass to self.model;
        - merge flat metric and loss dictionaries;
        - add the ML-step prefix before logging.

    Conventions:
        - metric sets return flat, unprefixed log dictionaries;
        - loss objects return flat, unprefixed log dictionaries;
        - this module adds the prefix corresponding to the ML step:
            train/, val/, or test/.

    This class does not:
        - define the model architecture;
        - build metric sets or losses;
        - compute metrics directly;
        - decide which metrics are differentiable;
        - implement task-specific training, validation, or test logic.

    Subclasses are expected to define:
        - self.model;
        - training_step;
        - validation_step;
        - test_step;
        - configure_optimizers.
    """

    valid_ml_steps = {"train", "val", "test"}

    def __init__(self, 
                 models : dict,
                 optimizer_configs : dict,
                 scheduler_configs : dict,
                 strict : bool = True):
        super().__init__()
        # self.save_hyperparameters()

        self.models = nn.ModuleDict(models)
        self.optimizer_configs = optimizer_configs
        self.scheduler_configs = scheduler_configs
        self.strict = strict

    def forward(self, x):
        raise NotImplementedError

    def training_step(self, batch, batch_idx: int,):
        raise NotImplementedError

    def validation_step(self, batch, batch_idx: int,):
        raise NotImplementedError

    def test_step(self, batch, batch_idx: int,):
        raise NotImplementedError

    

    def configure_optimizers(self):
        optimizers = build_optimizers_from_models(
            models=self.models,
            optimizer_configs=self.optimizer_configs,
            strict=self.strict
            )
        schedulers = build_schedulers(
            optimizer_groups=optimizers,
            scheduler_configs=self.scheduler_configs,
            strict=self.strict
            )

        optimizers = list(optimizers.values())
        schedulers = list(schedulers.values())

        if len(schedulers) == 0:
            return optimizers

        return optimizers, schedulers

    ####################################
    # Log helpers
    ####################################

    def _prepare_for_log(
        self,
        ml_step: str,
        *log_dicts: dict[str, object],
    ) -> dict[str, object]:
        """
        Merge flat log dictionaries and add the ML-step prefix.

        Example:
            {"loss": loss, "sisdr": sisdr}
            -> {"train/loss": loss, "train/sisdr": sisdr}
        """
        self.check_ml_step(ml_step)

        log_dict = self._merge_log_dicts(*log_dicts)
        return self._add_log_prefix(
            log_dict=log_dict,
            prefix=f"{ml_step}/",
        )

    def _merge_log_dicts(
        self,
        *log_dicts: dict[str, object],
    ) -> dict[str, object]:
        merged_log_dict = {}

        for log_dict in log_dicts:
            self.check_log_dict(log_dict)

            for key, value in log_dict.items():
                if key in merged_log_dict:
                    raise ValueError(
                        f"Duplicate log key before prefixing: {key}."
                    )

                merged_log_dict[key] = value

        return merged_log_dict

    def _add_log_prefix(
        self,
        log_dict: dict[str, object],
        prefix: str,
    ) -> dict[str, object]:
        return {
            f"{prefix}{key}": value
            for key, value in log_dict.items()
        }
    
    def log_step_dict(
        self,
        ml_step: str,
        *log_dicts: dict[str, object],
        **log_kwargs,
    ) -> None:
        log_dict = self._prepare_for_log(ml_step, *log_dicts)
        self.log_dict(log_dict, **log_kwargs)

    ####################################
    # Tests
    ####################################

    def check_ml_step(self, ml_step: str) -> None:
        if ml_step not in self.valid_ml_steps:
            raise ValueError(
                f"Invalid ML step: {ml_step}. "
                f"Expected one of {self.valid_ml_steps}."
            )

    def check_log_dict(self, log_dict: dict[str, object]) -> None:
        if not isinstance(log_dict, dict):
            raise TypeError(
                f"Expected log_dict to be a dictionary, got {type(log_dict)}."
            )

        for key in log_dict:
            if not isinstance(key, str):
                raise TypeError(
                    f"Log keys must be strings, got {type(key)}."
                )
            


