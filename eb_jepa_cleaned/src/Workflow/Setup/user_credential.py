"""
User credential loading.

Loads a private credential file (e.g. `user_credential.yaml`, gitignored) and
exports selected entries to environment variables — so the wandb API key (and
similar secrets) is settled from config during setup, before wandb login reads
`WANDB_API_KEY` from the environment.

Config shape:

    user_credential:
      enabled: true
      path: user_credential.yaml          # relative to project_root, or absolute
      environment_variables:              # env var name -> nested key path
        WANDB_API_KEY: [wandb, api_key]

Strict: when enabled, a missing path / file / key raises. The returned context
records what happened (enabled, path, loaded, exported env var names) but never
the secret values.
"""

import os
from pathlib import Path

from ..Configs.conversion import load_config


def setup_user_credential(
    config: dict | None = None,
    project_root: str | Path | None = None,
) -> dict:
    config = dict(config or {})
    if not config.get("enabled", False):
        return _context(enabled=False, path=None, loaded=False, exported=[])

    path = config.get("path")
    if not path:
        raise ValueError("user_credential is enabled but no 'path' was provided.")

    credential_path = _resolve_credential_path(path, project_root)
    if not credential_path.is_file():
        raise FileNotFoundError(f"User credential file not found: {credential_path}")

    credentials = load_config(credential_path)
    exported = _export_environment_variables(
        credentials=credentials,
        environment_variables=config.get("environment_variables", {}),
    )
    return _context(
        enabled=True, path=str(credential_path), loaded=True, exported=exported
    )


def _export_environment_variables(
    credentials: dict,
    environment_variables: dict[str, list[str]],
) -> list[str]:
    exported = []
    for env_var_name, key_path in environment_variables.items():
        value = _get_nested(credentials, key_path)
        os.environ[env_var_name] = str(value)
        exported.append(env_var_name)
    return sorted(exported)


def _get_nested(credentials: dict, key_path):
    if not isinstance(key_path, (list, tuple)):
        raise TypeError(
            f"Credential key path must be a list of keys, got {type(key_path)}."
        )

    current = credentials
    for key in key_path:
        if not isinstance(current, dict) or key not in current:
            raise KeyError(f"Missing credential key path {list(key_path)}.")
        current = current[key]
    return current


def _resolve_credential_path(path, project_root) -> Path:
    path = Path(path).expanduser()
    if path.is_absolute():
        return path.resolve()
    root = Path(project_root).expanduser().resolve() if project_root else Path.cwd()
    return (root / path).resolve()


def _context(enabled: bool, path, loaded: bool, exported: list[str]) -> dict:
    """Serializable credential context — never includes secret values."""
    return {
        "enabled": enabled,
        "path": path,
        "loaded": loaded,
        "exported_env_vars": exported,
    }
