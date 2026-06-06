from .base import (
    BaseLightningModule, 
)

from ..Metrics.MetricSet import TSEMetricSet
from ..Metrics.loss import WeightedMetricLoss

import torch


####################################
# TSE base lightning module
####################################


class TSEBaseLightningModule(BaseLightningModule):
    """
    Base Lightning module for Target Speech Extraction experiments.

    This class centralizes TSE-level metric and loss orchestration.

    Responsibilities:
        - store one TSE metric set per ML step;
        - select the correct metric set from the ML step;
        - compute flat TSE metric dictionaries from preds, target, mixture, clue;
        - compute the training loss from the train metric dictionary;
        - log metric and loss dictionaries through BaseLightningModule helpers.

    This class does not:
        - parse batches;
        - call the model;
        - define forward, training_step, validation_step, or test_step;
        - configure optimizers;
        - decide logger backends.
    """

    def __init__(
        self,
        train_metrics: TSEMetricSet,
        val_metrics: TSEMetricSet,
        test_metrics: TSEMetricSet,
        loss: WeightedMetricLoss,
        model_name : str,
        log_loss_ml_steps: tuple[str, ...] = ("train", "val"),
        **kwargs,
    ):
        """
        Initialize the TSE metric/loss orchestration layer.

        This __init__ is written to cooperate with mixins.

        In particular, **kwargs is intentionally kept so that other parent
        classes in the method resolution order can receive their own arguments.
        For example, a SingleModelOptimizerMixin may consume model and optimizer
        arguments before forwarding the remaining arguments here.

        Convention:
            every parent in the inheritance chain should:
                1. consume only the arguments it owns;
                2. forward the remaining arguments with super().__init__(**kwargs).

        The call to super().__init__(**kwargs) must happen before assigning
        TSEMetricSet and WeightedMetricLoss objects to self, because they are
        nn.Module instances and PyTorch requires nn.Module.__init__ to be called
        before registering submodules.
        """
        super().__init__(**kwargs)

        # One metric set per ML step.
        self.train_metrics = train_metrics
        self.val_metrics = val_metrics
        self.test_metrics = test_metrics

        # Loss aggregation object.
        # It consumes the flat metric dictionary produced by the selected metric set.
        self.loss = loss

        # Specify ML steps for which the loss is computed and logged.
        self.log_loss_ml_steps = log_loss_ml_steps

        # Classic TSE assumes a single model framework
        self.model = self.models[model_name]



    ####################################
    # TSE step processing helpers
    ####################################

    def process_tse_step_outputs(
        self,
        ml_step: str,
        preds: torch.Tensor,
        target: torch.Tensor,
        mixture: torch.Tensor | None = None,
        clue=None,
        **log_kwargs,
    ) -> torch.Tensor | None:
        """
        Compute and log TSE metrics and, for training, the loss.

        Convention:
            - train: logs metrics and loss terms, returns the loss;
            - val/test: logs metrics only, returns None.

        This method expects preds, target, mixture, and clue to already be
        extracted from the batch by the child module.
        """
        metric_logs = self.compute_tse_metrics(
            ml_step=ml_step,
            preds=preds,
            target=target,
            mixture=mixture,
            clue=clue,
        )

        loss = None
        loss_logs = {}

        if ml_step in self.log_loss_ml_steps:
            loss, loss_logs = self.compute_tse_loss(metric_logs)

        self.log_step_dict(ml_step, metric_logs, loss_logs, **log_kwargs)

        return loss

    def compute_tse_metrics(
        self,
        ml_step: str,
        preds: torch.Tensor,
        target: torch.Tensor,
        mixture: torch.Tensor | None = None,
        clue=None,
    ) -> dict[str, object]:
        metric_set = self.get_tse_metric_set(ml_step)

        return metric_set(
            preds=preds,
            target=target,
            mixture=mixture,
            clue=clue,
        )

    def compute_tse_loss(
        self,
        metric_logs: dict[str, object],
    ) -> tuple[torch.Tensor, dict[str, object]]:
        return self.loss(metric_logs)

    ####################################
    # Helpers
    ####################################

    def get_tse_metric_set(
        self,
        ml_step: str,
    ) -> TSEMetricSet:
        self.check_ml_step(ml_step)

        if ml_step == "train":
            return self.train_metrics

        if ml_step == "val":
            return self.val_metrics

        if ml_step == "test":
            return self.test_metrics

        # Unreachable if check_ml_step is consistent with valid_ml_steps.
        raise RuntimeError(f"Unhandled ML step: {ml_step}.")
    

    def __str__(self) -> str:
        mes = "TSESingleModelLightningModule with:\n\n"

        mes += "####################\n"
        mes += "# Model\n"
        mes += "####################\n"
        mes += f"{self.model}\n\n"

        mes += "####################\n"
        mes += "# Train metrics\n"
        mes += "####################\n"
        mes += f"{self.train_metrics}\n\n"

        mes += "####################\n"
        mes += "# Validation metrics\n"
        mes += "####################\n"
        mes += f"{self.val_metrics}\n\n"

        mes += "####################\n"
        mes += "# Test metrics\n"
        mes += "####################\n"
        mes += f"{self.test_metrics}\n\n"

        mes += "####################\n"
        mes += "# Loss\n"
        mes += "####################\n"
        mes += f"{self.loss}\n\n"

        mes += "####################\n"
        mes += "# Optimizer config\n"
        mes += "####################\n"
        mes += f"{self.optimizer_configs}\n\n"

        mes += "####################\n"
        mes += "# Scheduler config\n"
        mes += "####################\n"
        mes += f"{self.scheduler_configs}\n"

        return mes
    

