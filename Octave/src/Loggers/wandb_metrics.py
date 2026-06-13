from numbers import Number

import torch
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.loggers import WandbLogger


class WandbScalarMetricsCallback(Callback):
    """
    Make W&B scalar metric logging explicit for Lightning metrics.
    """

    def __init__(
        self,
        define_metrics: bool = True,
        direct_log: bool = True,
        step_metric: str = "trainer/global_step",
        metric_patterns: list[str] | None = None,
        require_wandb_logger: bool = True,
    ) -> None:
        self.define_metrics = define_metrics
        self.direct_log = direct_log
        self.step_metric = step_metric
        self.metric_patterns = metric_patterns or ["train/*", "val/*", "test/*"]
        self.require_wandb_logger = require_wandb_logger

    def on_fit_start(self, trainer, pl_module) -> None:
        wandb_loggers = get_wandb_loggers(trainer)
        self.check_wandb_loggers(wandb_loggers)

        if self.define_metrics:
            self.define_wandb_metrics(wandb_loggers)

    def on_train_batch_end(
        self,
        trainer,
        pl_module,
        outputs,
        batch,
        batch_idx: int,
    ) -> None:
        if self.direct_log:
            self.log_scalar_metrics(trainer=trainer, prefixes=("train/",))

    def on_validation_epoch_end(self, trainer, pl_module) -> None:
        if self.direct_log:
            self.log_scalar_metrics(trainer=trainer, prefixes=("val/",))

    def on_test_epoch_end(self, trainer, pl_module) -> None:
        if self.direct_log:
            self.log_scalar_metrics(trainer=trainer, prefixes=("test/",))

    def check_wandb_loggers(self, wandb_loggers: list[WandbLogger]) -> None:
        if self.require_wandb_logger and not wandb_loggers:
            raise RuntimeError(
                "W&B metrics were configured, but no WandbLogger is attached "
                "to the Lightning Trainer."
            )

    def define_wandb_metrics(self, wandb_loggers: list[WandbLogger]) -> None:
        for logger in wandb_loggers:
            experiment = logger.experiment
            define_metric = getattr(experiment, "define_metric", None)

            if define_metric is None:
                continue

            define_metric(self.step_metric)
            for pattern in self.metric_patterns:
                define_metric(pattern, step_metric=self.step_metric)

    def log_scalar_metrics(self, trainer, prefixes: tuple[str, ...]) -> None:
        metrics = collect_prefixed_scalar_metrics(
            trainer=trainer,
            prefixes=prefixes,
        )

        if not metrics:
            return

        metrics[self.step_metric] = trainer.global_step

        for logger in get_wandb_loggers(trainer):
            logger.experiment.log(metrics)


def get_wandb_loggers(trainer) -> list[WandbLogger]:
    loggers = getattr(trainer, "loggers", None)

    if loggers is None:
        logger = getattr(trainer, "logger", None)
        loggers = [] if logger is None or logger is False else [logger]

    return [
        logger
        for logger in loggers
        if isinstance(logger, WandbLogger)
    ]


def collect_prefixed_scalar_metrics(
    trainer,
    prefixes: tuple[str, ...],
) -> dict[str, float]:
    metrics = {}

    for metric_source_name in ("logged_metrics", "callback_metrics"):
        metric_source = getattr(trainer, metric_source_name, {})
        for key, value in metric_source.items():
            if not key.startswith(prefixes):
                continue

            scalar = to_python_scalar(value)
            if scalar is not None:
                metrics[key] = scalar

    return metrics


def to_python_scalar(value) -> float | int | None:
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            return None

        return value.detach().cpu().item()

    if isinstance(value, Number):
        return value

    return None
