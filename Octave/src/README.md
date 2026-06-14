# Octave Source README

This folder is the main source tree for Octave's clean AcVideoJepa
implementation.

## Folder Roles

`Data/`

- dataset wrappers;
- collators;
- datamodules;
- data transforms.

`Models/`

- AcVideoJepa architecture component construction;
- Lightning JEPA composition and orchestration;
- validation/evaluation weight loading.

`Rollouts/`

- latent rollout behavior over the JEPA Lightning runtime;
- autoregressive and parallel prediction policies;
- loss-free planning and inference trajectories.

`Metrics/`

- objective, loss, regularizer, and metric composition;
- thin wrappers around reusable EB-JEPA metric implementations;
- flat logging dictionaries for Lightning modules.

`Loggers/`

- CSV and wandb logger construction;
- wandb dashboards are supported;
- wandb sweeps are out of scope.

`Setup/`

- config loading and merging;
- runtime path resolution;
- reproducibility setup;
- wandb mode/login registration.

`Training/`

- optimizer construction;
- Lightning checkpoint callback construction;
- checkpoint path resolution for callbacks.

`Execution/`

- training and validation run orchestration;
- execution reports;
- external-service cleanup.

## Subsystem Contract

Source code in this tree should follow the Template architecture:

- configs stay plain and serializable;
- factories build objects from configs;
- runtime-dependent resolved information belongs in `runtime_context`;
- Lightning modules receive already-built objects;
- hidden global state is avoided;
- path resolution does not happen inside datasets, models, or modules;
- failures should be early and semantic.

## AcVideoJepa Boundary

The EB-JEPA package remains the implementation backend for reusable neural
network and loss primitives. Octave owns how those primitives are composed.

Octave should import and wrap existing EB-JEPA implementations instead of
recoding them:

- encoders, action encoders, and predictors are architecture components;
- rollout behavior belongs in `Rollouts/`;
- prediction losses, regularizers, and metric aggregation belong in `Metrics/`;
- `AcVideoJepaModule` is the Lightning-level JEPA runtime object.

The monolithic EB-JEPA `JEPA` wrapper is not an Octave runtime dependency.
Octave may match its conceptual behavior, but subsystem ownership is defined
here.

## Extension Steps

1. Add the source code in the owning subsystem.
2. Add or update that subsystem README.
3. Add a focused unit or integration test under `Octave/tests`.
4. Keep changes scoped to AcVideoJepa.
