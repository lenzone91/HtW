"""Small training primitives for JEPA world models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from Enzo.jepa_templates.specs import JEPABatchSpec, normalize_jepa_batch


@dataclass(frozen=True)
class TrainStepResult:
    """Output of one JEPA optimization step."""

    loss: Any
    metrics: dict[str, float]
    predictions: Any


def train_ac_jepa_step(
    jepa: Any,
    batch: dict[str, Any],
    optimizer: Any,
    scaler: Any,
    device: Any,
    nsteps: int,
    batch_spec: JEPABatchSpec = JEPABatchSpec(),
    scheduler: Any | None = None,
    use_amp: bool = True,
    dtype: Any | None = None,
    grad_clip: float | None = None,
) -> TrainStepResult:
    """Run one action-conditioned JEPA training step.

    This is intentionally probe-free. Add task probes in the calling script so
    the world-model loss remains easy to reuse across environments.
    """
    import torch
    from torch.amp import autocast

    normalized = normalize_jepa_batch(batch, batch_spec, device=device)
    if normalized.actions is not None and normalized.actions.shape[2] != normalized.obs.shape[2]:
        raise ValueError(
            "IDM-regularized AC-JEPA training expects actions and observations "
            "to share the same time length: "
            f"actions T={normalized.actions.shape[2]} vs obs T={normalized.obs.shape[2]}."
        )

    optimizer.zero_grad(set_to_none=True)
    with autocast(device.type, enabled=use_amp, dtype=dtype):
        predictions, losses = jepa.unroll(
            normalized.obs,
            normalized.actions,
            nsteps=nsteps,
            unroll_mode="autoregressive",
            ctxt_window_time=1,
            compute_loss=True,
            return_all_steps=False,
        )
        loss, reg_loss, reg_loss_unweighted, reg_loss_dict, pred_loss = losses

    if scaler is not None:
        scaler.scale(loss).backward()
        if grad_clip is not None:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(jepa.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()
    else:
        loss.backward()
        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(jepa.parameters(), grad_clip)
        optimizer.step()

    if scheduler is not None:
        scheduler.step()

    metrics = {
        "train/loss": float(loss.detach().item()),
        "train/reg_loss": float(reg_loss.detach().item()),
        "train/reg_loss_unweighted": float(reg_loss_unweighted.detach().item()),
        "train/pred_loss": float(pred_loss.detach().item()),
    }
    for name, value in reg_loss_dict.items():
        metrics[f"train/reg/{name}"] = float(value)

    return TrainStepResult(loss=loss, metrics=metrics, predictions=predictions)
