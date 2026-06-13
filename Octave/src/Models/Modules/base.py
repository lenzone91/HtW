import lightning.pytorch as pl


class BaseLightningModule(pl.LightningModule):
    """
    Shared LightningModule helpers for Octave modules.
    """

    valid_ml_steps = {"train", "val", "test"}

    def __init__(self, strict: bool = True) -> None:
        super().__init__()
        self.strict = strict

    def prepare_log_dict(
        self,
        ml_step: str,
        *log_dicts: dict[str, object],
    ) -> dict[str, object]:
        self.check_ml_step(ml_step)
        merged = self.merge_log_dicts(*log_dicts)
        return {
            f"{ml_step}/{key}": value
            for key, value in merged.items()
        }

    def log_step_dict(
        self,
        ml_step: str,
        *log_dicts: dict[str, object],
        **log_kwargs,
    ) -> None:
        log_dict = self.prepare_log_dict(ml_step, *log_dicts)

        if getattr(self, "_trainer", None) is None:
            return

        self.log_dict(log_dict, **log_kwargs)

    def merge_log_dicts(self, *log_dicts: dict[str, object]) -> dict[str, object]:
        merged = {}

        for log_dict in log_dicts:
            self.check_log_dict(log_dict)

            for key, value in log_dict.items():
                if key in merged:
                    raise ValueError(f"Duplicate log key before prefixing: {key}.")

                merged[key] = value

        return merged

    def check_ml_step(self, ml_step: str) -> None:
        if ml_step not in self.valid_ml_steps:
            raise ValueError(
                f"Invalid ML step: {ml_step}. "
                f"Expected one of {sorted(self.valid_ml_steps)}."
            )

    def check_log_dict(self, log_dict: dict[str, object]) -> None:
        if not isinstance(log_dict, dict):
            raise TypeError(
                f"Expected log_dict to be a dictionary, got {type(log_dict).__name__}."
            )

        for key in log_dict:
            if not isinstance(key, str):
                raise TypeError(f"Log keys must be strings, got {type(key).__name__}.")
