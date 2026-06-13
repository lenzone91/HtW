import os

import yaml

from Octave.src.Setup.setup import setup_runtime


def test_setup_runtime_loads_user_credentials_before_logger_registration(
    tmp_path,
    monkeypatch,
):
    project_root = tmp_path
    credential_path = project_root / "user_credential.yaml"
    config_path = project_root / "configs" / "runs" / "dummy.yaml"

    config_path.parent.mkdir(parents=True)
    config_path.write_text("{}", encoding="utf-8")
    credential_path.write_text(
        yaml.safe_dump({"WANDB_API_KEY": "fake-key"}),
        encoding="utf-8",
    )

    monkeypatch.delenv("WANDB_API_KEY", raising=False)

    setup_config = {
        "paths": {
            "project_root": str(project_root),
        },
        "logger_registration": {
            "wandb": {
                "enabled": False,
            },
        },
    }

    runtime_context = setup_runtime(
        setup_config=setup_config,
        config_path=config_path,
    )

    assert os.environ["WANDB_API_KEY"] == "fake-key"
    assert runtime_context["credentials"]["credential_path"] == str(credential_path)
    assert runtime_context["credentials"]["loaded_env_vars"] == ["WANDB_API_KEY"]