"""
Unit tests for AIML.Training.Loggers.factory.
"""

import pytest
from lightning.pytorch.loggers import CSVLogger as LightningCSVLogger

from src.AIML.Training.Loggers.factory import (
    build_loggers,
)
from src.Workflow.Factory.errors import (
    RegistryError,
)


def test_build_csv_logger(tmp_path):
    loggers = build_loggers({"csv": {"save_dir": str(tmp_path)}})

    assert isinstance(loggers, list)
    assert len(loggers) == 1
    assert isinstance(loggers[0], LightningCSVLogger)


def test_empty_config_disables_logging():
    assert build_loggers({}) is False


def test_unknown_logger_raises():
    with pytest.raises(RegistryError, match="Unknown logger"):
        build_loggers({"mystery": {}})
