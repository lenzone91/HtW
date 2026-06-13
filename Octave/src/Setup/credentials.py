import os
from pathlib import Path

import yaml


def load_user_credentials_to_env(
    credential_path: str | Path,
    *,
    override_existing: bool = False,
) -> dict[str, str]:
    """
    Load user credentials from a YAML file into os.environ.

    Expected YAML format:
        ENV_VAR_NAME: "secret_value"

    Existing environment variables are preserved by default.
    """
    credential_path = Path(credential_path).expanduser().resolve()

    if not credential_path.is_file():
        raise FileNotFoundError(f"Credential file not found: {credential_path}")

    with credential_path.open("r", encoding="utf-8") as file:
        credentials = yaml.safe_load(file)

    if not isinstance(credentials, dict):
        raise TypeError(
            f"Credential file must contain a dictionary, "
            f"got {type(credentials).__name__}."
        )

    loaded_credentials: dict[str, str] = {}

    for key, value in credentials.items():
        if not isinstance(key, str) or key.strip() == "":
            raise TypeError(
                "Credential keys must be non-empty strings. "
                f"Got key {key!r} of type {type(key).__name__}."
            )

        if not isinstance(value, str) or value.strip() == "":
            raise TypeError(
                f"Credential value for {key!r} must be a non-empty string, "
                f"got {type(value).__name__}."
            )

        if override_existing or key not in os.environ:
            os.environ[key] = value

        loaded_credentials[key] = os.environ[key]

    return loaded_credentials