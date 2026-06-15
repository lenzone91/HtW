from pathlib import Path

import pytest
import torch

from Octave.src.Models.Modules.ac_video_jepa_module import AcVideoJepaModule
from Octave.tests.unit.test_ac_video_jepa_module import make_module
from Octave.src.Models.Loading.factory import (
    load_module_if_needed,
)
from Octave.src.Models.Loading.module_loading import (
    load_module_from_lightning_checkpoint,
    resolve_checkpoint_path,
)


def make_runtime_context(tmp_path: Path) -> dict:
    return {
        "paths": {
            "run_dir": str(tmp_path / "run"),
        },
    }


def test_resolve_checkpoint_path_resolves_relative_path(tmp_path: Path) -> None:
    runtime_context = make_runtime_context(tmp_path)

    resolved_path = resolve_checkpoint_path(
        config={
            "checkpoint_path": "checkpoints/last.ckpt",
            "relative_to": "run_dir",
        },
        runtime_context=runtime_context,
    )

    assert resolved_path == str((tmp_path / "run" / "checkpoints" / "last.ckpt").resolve())


def test_load_module_if_needed_returns_module_when_disabled(tmp_path: Path) -> None:
    module = make_module()

    loaded_module = load_module_if_needed(
        module=module,
        loading_config={"enabled": False},
        runtime_context=make_runtime_context(tmp_path),
    )

    assert loaded_module is module


def test_load_module_from_lightning_checkpoint_loads_state_dict(tmp_path: Path) -> None:
    source_module = make_module()
    checkpoint_path = tmp_path / "module.ckpt"
    torch.save({"state_dict": source_module.state_dict()}, checkpoint_path)

    target_module = make_module()

    loaded_module = load_module_from_lightning_checkpoint(
        module=target_module,
        checkpoint_path=str(checkpoint_path),
    )

    assert loaded_module is target_module
    for key, value in source_module.state_dict().items():
        assert torch.equal(value, target_module.state_dict()[key])


def test_load_module_if_needed_loads_enabled_module(tmp_path: Path) -> None:
    source_module = make_module()
    checkpoint_dir = tmp_path / "run" / "checkpoints"
    checkpoint_dir.mkdir(parents=True)
    checkpoint_path = checkpoint_dir / "module.ckpt"
    torch.save({"state_dict": source_module.state_dict()}, checkpoint_path)

    target_module = make_module()

    loaded_module = load_module_if_needed(
        module=target_module,
        loading_config={
            "enabled": True,
            "type": "lightning_module",
            "checkpoint_path": "checkpoints/module.ckpt",
            "strict": True,
            "map_location": "cpu",
            "state_dict_key": "state_dict",
            "relative_to": "run_dir",
        },
        runtime_context=make_runtime_context(tmp_path),
    )

    assert isinstance(loaded_module, AcVideoJepaModule)
    for key, value in source_module.state_dict().items():
        assert torch.equal(value, target_module.state_dict()[key])


def test_load_module_if_needed_rejects_wrong_loading_type(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="Unknown loading"):
        load_module_if_needed(
            module=make_module(),
            loading_config={
                "enabled": True,
                "type": "torch_model",
                "checkpoint_path": "checkpoint.ckpt",
            },
            runtime_context=make_runtime_context(tmp_path),
        )
