# Model README

This folder owns architecture block construction for AcVideoJepa.

## Folder Roles

`ac_video_jepa/`

- builds AcVideoJepa encoder, action encoder, and predictor blocks;
- wraps EB-JEPA architecture implementations;
- exposes config-driven block factories.

## Ownership Rules

Model code owns architecture block construction.

Rollout code owns latent dynamics unrolling.

Metrics code owns losses, regularizers, and objective aggregation.

Lightning modules own JEPA orchestration.

Data code owns samples, batches, and DataLoaders.
