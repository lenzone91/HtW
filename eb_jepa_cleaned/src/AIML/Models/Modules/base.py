import lightning.pytorch as pl
from lightning.pytorch.loggers import WandbLogger
from torch import nn

from ...Training.Optimizers.factory import build_optimizers_from_models
from ...Training.Schedulers.factory import build_schedulers


# Gradient logging is a sanity check, not a metric: it goes to W&B (histograms
# via wandb.watch), not the CSV. Disabled by default; opt in via the
# `watch_gradients` module config.
DEFAULT_WATCH_GRADIENTS = {
    "enabled": False,
    "log": "gradients",  # "gradients" | "parameters" | "all"
    "log_freq": 100,     # log every N training steps
}


#############################################
# Base Lightning module
#############################################


class BaseLightningModule(pl.LightningModule):
    """
    Base PyTorch Lightning module.

    Centralizes Lightning-level utilities shared across task-specific modules:
        - holds the model dict;
        - builds optimizers/schedulers from configs in configure_optimizers;
        - merges flat metric/loss log dicts and adds the ML-step prefix.

    Conventions:
        - metric sets and losses return flat, unprefixed log dicts;
        - this module adds the `train/`, `val/`, or `test/` prefix when logging.

    It does NOT define the architecture, build metric sets or losses, or
    implement task-specific step logic. Subclasses define `self.model(s)`,
    `forward`, the `*_step` methods, and (optionally) override behavior.

    Note on optimizers/schedulers: these are received as CONFIGS (not built
    objects) because Lightning owns their lifecycle and builds them from the
    module's parameters inside configure_optimizers. Models, metrics, and losses
    are passed already built (Decision 15).
    """

    valid_ml_steps = {"train", "val", "test"}

    def __init__(
        self,
        models: dict,
        optimizer_configs: dict,
        scheduler_configs: dict,
        watch_gradients: dict | None = None,
    ):
        super().__init__()

        self.models = nn.ModuleDict(models)
        self.optimizer_configs = optimizer_configs
        self.scheduler_configs = scheduler_configs
        self.watch_gradients = {
            **DEFAULT_WATCH_GRADIENTS,
            **(watch_gradients or {}),
        }

    def forward(self, x):
        raise NotImplementedError

    def training_step(self, batch, batch_idx: int):
        raise NotImplementedError

    def validation_step(self, batch, batch_idx: int):
        raise NotImplementedError

    def test_step(self, batch, batch_idx: int):
        raise NotImplementedError

    def configure_optimizers(self):
        optimizers = build_optimizers_from_models(
            models=self.models,
            optimizer_configs=self.optimizer_configs,
        )
        schedulers = build_schedulers(
            optimizer_groups=optimizers,
            scheduler_configs=self.scheduler_configs,
        )

        optimizers = list(optimizers.values())
        schedulers = list(schedulers.values())

        if len(schedulers) == 0:
            return optimizers

        return optimizers, schedulers

    #############################################
    # Gradient logging (W&B sanity check)
    #############################################

    def _wandb_loggers(self) -> list:
        loggers = self.trainer.loggers if self.trainer else []
        return [logger for logger in loggers if isinstance(logger, WandbLogger)]

    def on_fit_start(self) -> None:
        """
        Ask each W&B logger to watch this module (gradient/parameter histograms).

        Opt in via the `watch_gradients` module config. This is a sanity check,
        so it only targets W&B; a no-op when no W&B logger is active.
        """
        config = getattr(self, "watch_gradients", DEFAULT_WATCH_GRADIENTS)
        if not config.get("enabled", False):
            return
        for logger in self._wandb_loggers():
            logger.watch(
                self,
                log=config.get("log", "gradients"),
                log_freq=config.get("log_freq", 100),
            )

    def on_fit_end(self) -> None:
        config = getattr(self, "watch_gradients", DEFAULT_WATCH_GRADIENTS)
        if not config.get("enabled", False):
            return
        for logger in self._wandb_loggers():
            # Remove the wandb hooks so a second fit in the same process does not
            # double-watch; defensive against backend/offline quirks.
            try:
                logger.experiment.unwatch(self)
            except Exception:
                pass

    #############################################
    # Log helpers
    #############################################

    def _prepare_for_log(
        self,
        ml_step: str,
        *log_dicts: dict[str, object],
    ) -> dict[str, object]:
        """
        Merge flat log dicts and add the ML-step prefix.

        Example: {"loss": ..., "sisdr": ...} -> {"train/loss": ..., "train/sisdr": ...}
        """
        self.check_ml_step(ml_step)

        log_dict = self._merge_log_dicts(*log_dicts)
        return self._add_log_prefix(log_dict=log_dict, prefix=f"{ml_step}/")

    def _merge_log_dicts(
        self,
        *log_dicts: dict[str, object],
    ) -> dict[str, object]:
        merged_log_dict = {}

        for log_dict in log_dicts:
            self.check_log_dict(log_dict)

            for key, value in log_dict.items():
                if key in merged_log_dict:
                    raise ValueError(f"Duplicate log key before prefixing: {key}.")

                merged_log_dict[key] = value

        return merged_log_dict

    def _add_log_prefix(
        self,
        log_dict: dict[str, object],
        prefix: str,
    ) -> dict[str, object]:
        return {f"{prefix}{key}": value for key, value in log_dict.items()}

    def log_step_dict(
        self,
        ml_step: str,
        *log_dicts: dict[str, object],
        **log_kwargs,
    ) -> None:
        log_dict = self._prepare_for_log(ml_step, *log_dicts)
        self.log_dict(log_dict, **log_kwargs)

    #############################################
    # Validation helpers
    #############################################

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
                raise TypeError(f"Log keys must be strings, got {type(key)}.")
