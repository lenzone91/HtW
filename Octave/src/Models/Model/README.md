# Model README

This folder owns architecture component construction for AcVideoJepa.

## Folder Roles

`ac_video_jepa/`

- builds AcVideoJepa encoder, action encoder, and predictor components;
- wraps EB-JEPA architecture implementations;
- exposes config-driven component factories.

## Ownership Rules

Model code owns architecture component construction.

Rollout code owns latent dynamics unrolling.

Metrics code owns losses, regularizers, and objective aggregation.

Lightning modules own JEPA orchestration.

Data code owns samples, batches, and DataLoaders.
