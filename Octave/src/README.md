# Octave Source README

This folder is the main source tree for the AcVideoJepa migration.

## Folder Roles

`Data/`

- dataset wrappers;
- collators;
- datamodules;
- data transforms.

`Models/`

- AcVideoJepa model construction;
- Lightning module orchestration;
- validation/evaluation weight loading.

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

## Extension Steps

1. Add the source code in the owning subsystem.
2. Add or update that subsystem README.
3. Add a focused unit or integration test under `Octave/tests`.
4. Keep changes scoped to AcVideoJepa.
