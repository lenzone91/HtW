import os

import pytest
import yaml

from Octave.src.Setup.credentials import load_user_credentials_to_env

def test_load_user_credentials_to_env_sets_missing_env_var(tmp_path, monkeypatch):
    credential_path = tmp_path / "user_credential.yaml"
    credential_path.write_text(
        yaml.safe_dump({"WANDB_API_KEY": "fake-key"}),
        encoding="utf-8",
    )

    monkeypatch.delenv("WANDB_API_KEY", raising=False)

    loaded_credentials = load_user_credentials_to_env(credential_path)

    assert os.environ["WANDB_API_KEY"] == "fake-key"
    assert loaded_credentials == {"WANDB_API_KEY": "fake-key"}


def test_load_user_credentials_to_env_does_not_override_existing_env_var_by_default(
    tmp_path,
    monkeypatch,
):
    credential_path = tmp_path / "user_credential.yaml"
    credential_path.write_text(
        yaml.safe_dump({"WANDB_API_KEY": "file-key"}),
        encoding="utf-8",
    )

    monkeypatch.setenv("WANDB_API_KEY", "existing-key")

    loaded_credentials = load_user_credentials_to_env(credential_path)

    assert os.environ["WANDB_API_KEY"] == "existing-key"
    assert loaded_credentials == {"WANDB_API_KEY": "existing-key"}


def test_load_user_credentials_to_env_can_override_existing_env_var(
    tmp_path,
    monkeypatch,
):
    credential_path = tmp_path / "user_credential.yaml"
    credential_path.write_text(
        yaml.safe_dump({"WANDB_API_KEY": "file-key"}),
        encoding="utf-8",
    )

    monkeypatch.setenv("WANDB_API_KEY", "existing-key")

    loaded_credentials = load_user_credentials_to_env(
        credential_path,
        override_existing=True,
    )

    assert os.environ["WANDB_API_KEY"] == "file-key"
    assert loaded_credentials == {"WANDB_API_KEY": "file-key"}


def test_load_user_credentials_to_env_rejects_missing_file(tmp_path):
    credential_path = tmp_path / "missing_user_credential.yaml"

    with pytest.raises(FileNotFoundError):
        load_user_credentials_to_env(credential_path)


def test_load_user_credentials_to_env_rejects_non_dict_yaml(tmp_path):
    credential_path = tmp_path / "user_credential.yaml"
    credential_path.write_text(
        yaml.safe_dump(["WANDB_API_KEY", "fake-key"]),
        encoding="utf-8",
    )

    with pytest.raises(TypeError, match="must contain a dictionary"):
        load_user_credentials_to_env(credential_path)


def test_load_user_credentials_to_env_rejects_empty_key(tmp_path):
    credential_path = tmp_path / "user_credential.yaml"
    credential_path.write_text(
        yaml.safe_dump({"": "fake-key"}),
        encoding="utf-8",
    )

    with pytest.raises(TypeError, match="keys must be non-empty strings"):
        load_user_credentials_to_env(credential_path)


def test_load_user_credentials_to_env_rejects_empty_value(tmp_path):
    credential_path = tmp_path / "user_credential.yaml"
    credential_path.write_text(
        yaml.safe_dump({"WANDB_API_KEY": ""}),
        encoding="utf-8",
    )

    with pytest.raises(TypeError, match="must be a non-empty string"):
        load_user_credentials_to_env(credential_path)