from pathlib import Path

from .config_resolution import merge_configs
from .configs import DEFAULT_SETUP_CONFIG
from .logger_registration import setup_logger_registration
from .paths import setup_paths
from .reproducibility import setup_reproducibility


def setup_runtime(
    setup_config: dict | None = None,
    config_path: str | Path | None = None,
) -> dict:
    resolved_setup_config = merge_configs(
        base_config=DEFAULT_SETUP_CONFIG,
        override_config=setup_config or {},
        strict=False,
    )

    runtime_context = {}
    runtime_context["paths"] = setup_paths(
        paths_config=resolved_setup_config["paths"],
        config_path=config_path,
    )
    runtime_context["reproducibility"] = setup_reproducibility(
        resolved_setup_config["reproducibility"],
    )
    runtime_context["logger_registration"] = setup_logger_registration(
        resolved_setup_config["logger_registration"],
    )

    return runtime_context
