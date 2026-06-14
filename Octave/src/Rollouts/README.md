# Rollouts README

This folder owns loss-free rollout behavior for Octave runtime objects.

## Folder Role

`Rollouts/` converts a JEPA runtime object and batch tensors into predicted
latent trajectories.

For AcVideoJepa, rollout behavior reproduces the conceptual latent prediction
policy without depending on the monolithic EB-JEPA `JEPA` wrapper or owning loss
computation.

## File Roles

`latent_rollout.py`

- defines `LatentRollout`;
- defines the structured `LatentRolloutOutput`;
- implements parallel and autoregressive latent rollout behavior.

`configs.py`

- stores plain default rollout configs.

`factory.py`

- exposes public rollout construction wrappers;
- delegates config validation and type dispatch to `Workflow/Factory`.

`registry.py`

- owns the rollout registry and rollout builder declaration.

## Ownership Rules

Rollout code may:

- call an already-built encoder;
- call an already-built action encoder;
- call an already-built predictor;
- implement autoregressive and parallel latent prediction policies;
- validate rollout-specific config and tensor shape assumptions;
- return structured runtime outputs for objectives and planners.

Rollout code must not:

- build architecture components;
- instantiate losses, metrics, or regularizers;
- aggregate training objectives;
- parse dataloader batches beyond tensors it is explicitly given;
- resolve paths or read run YAML files.

## Extension Steps

1. Add plain rollout defaults in `configs.py`.
2. Register the rollout class with `ROLLOUT_REGISTRY`.
3. Keep public construction wrappers in `factory.py`.
4. Add unit tests for shapes, modes, and semantic failures.
5. Update this README when rollout behavior changes.
