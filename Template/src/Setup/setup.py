from pathlib import Path

from .environment import setup_environment
from .hardware import setup_hardware
from .logger_registration import setup_logger_registration
from .paths import setup_paths
from .project_root import resolve_project_root
from .reproducibility import setup_reproducibility
from .user_credential import setup_user_credential
from .data import setup_data


def setup_runtime(
    setup_config: dict | None = None,
    config_path: str | Path | None = None,
) -> dict:
    """
    Prepare the runtime environment and return runtime context.
    """
    if setup_config is None:
        setup_config = {}

    project_root = resolve_project_root(
        project_root=setup_config.get("paths", {}).get("project_root", None),
        config_path=config_path,
    )

    runtime_context = {}

    runtime_context["environment"] = setup_environment(
        setup_config.get("environment", {})
    )

    runtime_context["hardware"] = setup_hardware(
        setup_config.get("hardware", {})
    )

    runtime_context["reproducibility"] = setup_reproducibility(
        setup_config.get("reproducibility", {})
    )

    runtime_context["paths"] = setup_paths(
        paths_config=setup_config.get("paths", {}),
        project_root=project_root,
    )

    runtime_context["data"] = setup_data(
        data_config=setup_config.get("data", {}),
        project_root=runtime_context["paths"]["project_root"],
    )

    runtime_context["user_credential"] = setup_user_credential(
        user_credential_config=setup_config.get("user_credential", {}),
        project_root=runtime_context["paths"]["project_root"],
    )

    runtime_context["logger_registration"] = setup_logger_registration(
        setup_config.get("logger_registration", {})
    )

    return runtime_context