"""Runtime setup helpers for JEPA experiments."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from Enzo.jepa_templates.pathing import ensure_eb_jepa_importable


@dataclass
class ExperimentContext:
    """Shared runtime objects used by training scripts."""

    cfg: Any
    device: Any
    exp_dir: Path
    exp_name: str
    wandb_run: Any | None
    use_amp: bool
    dtype: Any
    scaler: Any


def load_cfg(fname: str | Path, overrides: dict[str, Any] | None = None) -> Any:
    """Load an OmegaConf config through EB-JEPA utilities."""
    ensure_eb_jepa_importable()
    from eb_jepa.training_utils import load_config

    return load_config(fname, overrides)


def setup_experiment(
    cfg: Any,
    example_name: str = "htw_ac_jepa",
    folder: str | Path | None = None,
    project: str = "htw_jepa",
    tags: list[str] | None = None,
) -> ExperimentContext:
    """Create the common runtime context for a JEPA training script."""
    ensure_eb_jepa_importable()

    import torch
    from omegaconf import OmegaConf
    from torch.amp import GradScaler

    from eb_jepa.training_utils import (
        get_default_dev_name,
        get_unified_experiment_dir,
        setup_device,
        setup_seed,
        setup_wandb,
    )

    device = setup_device(_cfg_get(cfg, "meta.device", "auto"))
    setup_seed(_cfg_get(cfg, "meta.seed", 1))

    exp_name = _cfg_get(cfg, "logging.exp_name", example_name)
    if folder is None:
        folder = _cfg_get(cfg, "meta.model_folder", None)

    if folder is None:
        exp_dir = get_unified_experiment_dir(
            example_name=example_name,
            sweep_name=_cfg_get(cfg, "logging.sweep_name", get_default_dev_name()),
            exp_name=exp_name,
            seed=_cfg_get(cfg, "meta.seed", 1),
        )
    else:
        exp_dir = Path(folder)
        exp_dir.mkdir(parents=True, exist_ok=True)

    use_amp = bool(_cfg_get(cfg, "training.use_amp", True))
    dtype_name = str(_cfg_get(cfg, "training.dtype", "bfloat16")).lower()
    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16}.get(
        dtype_name,
        torch.bfloat16,
    )
    scaler = GradScaler(device.type, enabled=use_amp and dtype == torch.float16)

    wandb_run = setup_wandb(
        project=project,
        config=OmegaConf.to_container(cfg, resolve=True),
        run_dir=exp_dir,
        run_name=exp_name,
        tags=tags or [example_name, f"seed_{_cfg_get(cfg, 'meta.seed', 1)}"],
        group=_cfg_get(cfg, "logging.wandb_group", None),
        enabled=bool(_cfg_get(cfg, "logging.log_wandb", False)),
        sweep_id=_cfg_get(cfg, "logging.wandb_sweep_id", None),
    )

    return ExperimentContext(
        cfg=cfg,
        device=device,
        exp_dir=Path(exp_dir),
        exp_name=exp_name,
        wandb_run=wandb_run,
        use_amp=use_amp,
        dtype=dtype,
        scaler=scaler,
    )


def _cfg_get(cfg: Any, dotted_key: str, default: Any = None) -> Any:
    current = cfg
    for key in dotted_key.split("."):
        if current is None:
            return default
        if isinstance(current, dict):
            current = current.get(key, default)
        else:
            current = getattr(current, key, default)
    return current

