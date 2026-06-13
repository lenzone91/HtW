# Models README

This folder owns model-side runtime objects for the AcVideoJepa migration.

## Folder Roles

`Model/`

- builds the AcVideoJepa world model from plain configs;
- wraps EB-JEPA architecture components.

`Modules/`

- owns LightningModule orchestration;
- receives already-built model objects.

`Loading/`

- loads weights into already-built modules for validation and evaluation.

## Subsystem Contract

Models code may:

- construct AcVideoJepa architecture objects;
- define Lightning training, validation, and test steps;
- load checkpoint weights into objects that were already built by factories.

Models code must not:

- resolve dataset paths;
- build dataloaders;
- configure Trainer callbacks;
- perform training resume.

## Extension Steps

1. Put architecture construction in `Model/`.
2. Put Lightning orchestration in `Modules/`.
3. Put validation/evaluation weight loading in `Loading/`.
4. Add focused tests under `Octave/tests`.
5. Update the local README for the touched subsystem.

## Ownership Rules

Model factories own architecture construction.

Lightning modules own step orchestration.

Execution owns when training or validation runs happen.
