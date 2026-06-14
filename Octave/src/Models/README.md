# Models README

This folder owns model-side runtime objects for Octave's AcVideoJepa
implementation.

## Folder Roles

`Model/`

- builds AcVideoJepa architecture components from plain configs;
- wraps EB-JEPA encoder, action encoder, and predictor components.

`Modules/`

- owns LightningModule JEPA orchestration;
- receives already-built components, rollouts, objectives, optimizers, and
  schedulers.

`Loading/`

- loads weights into already-built modules for validation and evaluation.

## Subsystem Contract

Models code may:

- construct AcVideoJepa architecture components;
- define Lightning training, validation, and test steps;
- load checkpoint weights into objects that were already built by factories.

Models code must not:

- resolve dataset paths;
- build dataloaders;
- own rollout algorithms;
- own prediction losses or regularizers;
- configure Trainer callbacks;
- perform training resume.

## Extension Steps

1. Put architecture component construction in `Model/`.
2. Put loss-free rollout behavior in `Rollouts/`.
3. Put objective and metric composition in `Metrics/`.
4. Put Lightning orchestration in `Modules/`.
5. Put validation/evaluation weight loading in `Loading/`.
6. Add focused tests under `Octave/tests`.
7. Update the local README for the touched subsystem.

## Ownership Rules

Model factories own architecture construction.

Rollout factories own rollout behavior.

Metrics factories own objective and loss composition.

Lightning modules own JEPA step orchestration.

Execution owns when training or validation runs happen.
