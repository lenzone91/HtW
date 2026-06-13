# Model README

This folder owns model construction for AcVideoJepa.

## Folder Roles

`ac_video_jepa/`

- builds the AcVideoJepa world model;
- wraps EB-JEPA model components;
- exposes config-driven model factories.

## Ownership Rules

Model code owns architecture construction.

Lightning modules own training-step orchestration.

Data code owns samples, batches, and DataLoaders.
