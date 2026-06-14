# Rollouts README

This folder owns loss-free rollout behavior for Octave runtime objects.

## Folder Role

`Rollouts/` converts already-built architecture blocks and batch tensors into
predicted latent trajectories.

For AcVideoJepa, rollout behavior is expected to wrap the algorithm currently
embedded in EB-JEPA `JEPA.unroll` without owning loss computation.

## File Roles

`latent_rollout.py`

- defines `LatentRollout`;
- defines the structured `LatentRolloutOutput`;
- implements parallel and autoregressive latent rollout behavior.

`configs.py`

- stores plain default rollout configs.

`factory.py`

- builds rollout objects from plain dictionaries;
- rejects unsupported rollout types and unknown config keys.

## Ownership Rules

Rollout code may:

- call an already-built encoder;
- call an already-built action encoder;
- call an already-built predictor;
- implement autoregressive and parallel latent prediction policies;
- validate rollout-specific config and tensor shape assumptions;
- return structured runtime outputs for objectives and planners.

Rollout code must not:

- build architecture blocks;
- instantiate losses, metrics, or regularizers;
- aggregate training objectives;
- parse dataloader batches beyond tensors it is explicitly given;
- resolve paths or read run YAML files.

## Extension Steps

1. Add plain rollout defaults in `configs.py`.
2. Add rollout construction in `factory.py`.
3. Add rollout implementation in a focused module.
4. Add unit tests for shapes, modes, and semantic failures.
5. Update this README when rollout behavior changes.
