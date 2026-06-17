"""Logger save_dir resolution against the runtime-context run paths."""

from src.AIML.Training.Loggers.factory import build_loggers
from src.AIML.Training.Loggers.loggers import (
    WandbLogger,
    resolve_logger_save_dir,
)


def test_resolver_uses_runtime_logs_dir():
    resolved = resolve_logger_save_dir(
        {"save_dir": "logs"}, {"paths": {"logs_dir": "/run/logs"}}
    )
    assert resolved == "/run/logs"


def test_resolver_falls_back_to_config_without_runtime():
    assert resolve_logger_save_dir({"save_dir": "mylogs"}, None) == "mylogs"


def test_build_wandb_logger_resolves_save_dir_to_run_logs(tmp_path):
    loggers = build_loggers(
        {"wandb": {"offline": True, "project": "t"}},
        runtime_context={"paths": {"logs_dir": str(tmp_path)}},
    )
    assert len(loggers) == 1
    assert isinstance(loggers[0], WandbLogger)
    assert str(loggers[0].save_dir) == str(tmp_path)


def test_empty_logger_config_disables_logging():
    assert build_loggers({}) is False
