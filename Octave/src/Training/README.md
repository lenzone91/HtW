# Training README

This folder owns training support factories for Octave.

## Folder Roles

`Optimization/`

- builds optimizers from plain config dictionaries;
- validates optimizer config keys.

`Schedulers/`

- builds learning-rate schedulers from plain config dictionaries;
- returns Lightning-compatible scheduler dictionaries.

`Checkpoints/`

- builds Lightning checkpoint callbacks;
- resolves callback directories through `runtime_context`.

## Subsystem Contract

Training support code may:

- build optimizers from parameters supplied by a caller;
- build schedulers from optimizers supplied by a caller;
- build checkpoint callbacks from execution configs;
- fail early on unknown config keys or unsupported types.

Training support code must not:

- build models or datamodules;
- run `trainer.fit` or `trainer.validate`;
- load model weights for validation;
- decide experiment paths outside `runtime_context`.

## Extension Steps

1. Add plain default configs in the owning subfolder.
2. Add factory logic in the owning subfolder.
3. Keep Trainer orchestration in `Execution/`.
4. Add focused unit tests.
5. Update the local README when contracts change.

## Ownership Rules

Optimization owns optimizer construction.

Schedulers own learning-rate scheduler construction.

Checkpoints own callback construction.

Execution owns resume semantics and run lifecycle.
