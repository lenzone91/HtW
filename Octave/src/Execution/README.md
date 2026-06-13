# Execution README

This folder owns train and validation run orchestration.

## File Roles

`factory.py`

- builds execution-ready objects;
- delegates DataModule construction to `Data/DataModules`;
- delegates Lightning module construction to `Models/Modules`;
- delegates logger construction to `Loggers`;
- delegates checkpoint callback construction to `Training/Checkpoints`;
- delegates validation/evaluation module weight loading to `Models/Loading`;
- instantiates the Lightning `Trainer`.

`train.py`

- runs `trainer.fit`;
- loads config files when called with `config_path`;
- calls `Setup/setup_runtime` when no runtime context is provided;
- creates and saves a training execution report.

`validate.py`

- runs `trainer.validate`;
- loads config files when called with `config_path`;
- calls `Setup/setup_runtime` when no runtime context is provided;
- creates and saves a validation execution report.

`reports.py`

- creates JSON-serializable execution reports;
- stores status, outputs, errors, and runtime context.

`cleanup.py`

- closes optional external services such as wandb.

`configs.py`

- stores plain execution-level defaults.

## Subsystem Contract

Execution may:

- request runtime setup;
- assemble already-built subsystem objects;
- launch Lightning trainer methods;
- pass checkpoint callbacks into the Trainer;
- pass `ckpt_path` into `trainer.fit` for resume;
- load module weights before validation;
- enable wandb parameter and gradient watching through configured loggers;
- save execution reports;
- close external services.

Execution must not:

- build model internals directly;
- build dataset internals directly;
- implement metrics;
- implement wandb sweeps.

## Wandb Scope

Wandb logging is supported through `Loggers/`. Execution applies configured
wandb `watch` settings after the module and Trainer loggers are built.

Wandb sweeps are not part of this project.

## Resume Contract

Training resume is configured with:

```yaml
resume:
  enabled: true
  checkpoint_path: path/to/checkpoint.ckpt
```

Execution passes this path to Lightning as `trainer.fit(..., ckpt_path=...)`.

Validation/evaluation loading is separate:

```yaml
loading:
  module:
    enabled: true
    checkpoint_path: path/to/checkpoint.ckpt
```

Loading restores module weights into an already-built module before validation.
It does not restore optimizer, scheduler, or Trainer loop state.
