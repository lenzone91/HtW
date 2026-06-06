"""Batch contracts shared by JEPA templates."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


@dataclass(frozen=True)
class JEPABatchSpec:
    """Keys used to read a world-model batch.

    Expected tensors:
      obs: [B, C, T, H, W]
      actions: [B, A, T_actions] or None
        For IDM-regularized training, T_actions must match the observation T.
      probe_target: optional task-specific supervision for probes
    """

    obs: str = "obs"
    actions: str = "actions"
    probe_target: str = "probe_target"
    meta: str = "meta"


@dataclass(frozen=True)
class JEPABatch:
    """Normalized JEPA batch object."""

    obs: Any
    actions: Any | None = None
    probe_target: Any | None = None
    meta: Any | None = None


def normalize_jepa_batch(
    batch: Mapping[str, Any],
    spec: JEPABatchSpec = JEPABatchSpec(),
    device: Any | None = None,
    non_blocking: bool = True,
) -> JEPABatch:
    """Read and optionally move a batch following the JEPA batch contract."""
    obs = _required(batch, spec.obs)
    actions = batch.get(spec.actions)
    probe_target = batch.get(spec.probe_target)
    meta = batch.get(spec.meta)

    if device is not None:
        obs = _move_to_device(obs, device, non_blocking)
        actions = _move_to_device(actions, device, non_blocking)
        probe_target = _move_to_device(probe_target, device, non_blocking)

    normalized = JEPABatch(
        obs=obs,
        actions=actions,
        probe_target=probe_target,
        meta=meta,
    )
    validate_jepa_batch(normalized)
    return normalized


def validate_jepa_batch(batch: JEPABatch) -> None:
    """Validate the minimum tensor shapes expected by EB-JEPA."""
    if not hasattr(batch.obs, "ndim"):
        raise TypeError("batch.obs must be a tensor-like object with an ndim attribute.")
    if batch.obs.ndim != 5:
        raise ValueError(f"batch.obs must be [B, C, T, H, W], got ndim={batch.obs.ndim}.")

    if batch.actions is not None:
        if not hasattr(batch.actions, "ndim"):
            raise TypeError("batch.actions must be tensor-like when provided.")
        if batch.actions.ndim != 3:
            raise ValueError(
                "batch.actions must be [B, A, T_actions], "
                f"got ndim={batch.actions.ndim}."
            )
        if batch.actions.shape[0] != batch.obs.shape[0]:
            raise ValueError(
                "batch.actions and batch.obs must have the same batch size: "
                f"{batch.actions.shape[0]} != {batch.obs.shape[0]}."
            )


def _required(batch: Mapping[str, Any], key: str) -> Any:
    if key not in batch:
        raise KeyError(f"Missing required JEPA batch key: {key!r}.")
    return batch[key]


def _move_to_device(value: Any, device: Any, non_blocking: bool) -> Any:
    if value is None:
        return None
    if hasattr(value, "to"):
        return value.to(device, non_blocking=non_blocking)
    return value
