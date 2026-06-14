# Execution README

This folder owns training and validation orchestration for Octave.

Execution wires together already-migrated subsystem factories:

```text
config + runtime_context -> runtime objects -> train/validate run
```

It does not own dataset behavior, model behavior, metric behavior, checkpoint behavior, or logger behavior.

## File Roles

```text
Execution/
- train.py
- validate.py
- factory.py
- trainers.py
- registry.py
- configs.py
- cleanup.py
- reports.py
- README.md
```

`train.py`

* exposes `run_training`;
* resolves the execution config;
* creates `runtime_context` through Setup when needed;
* builds training objects;
* calls `trainer.fit(...)`;
* writes an execution report;
* closes external services.

`validate.py`

* exposes `run_validation`;
* resolves the execution config;
* creates `runtime_context` through Setup when needed;
* builds validation objects;
* calls `trainer.validate(...)`;
* writes an execution report;
* closes external services.

`factory.py`

* builds execution-time object bundles;
* builds datamodules through `Data/DataModules`;
* builds Lightning modules through `Models/Modules`;
* builds loggers through `Loggers`;
* builds checkpoint callbacks through `Training/Checkpoints`;
* builds logger callbacks through `Loggers`;
* builds Trainer objects through the registered Trainer factory.

`trainers.py`

* defines the registered Lightning Trainer adapter;
* registers Trainer construction through the decorator-based factory API.

`registry.py` defines:

```text
TRAINER_REGISTRY
TRAINER_BUILDER
```

`configs.py` stores plain execution and Trainer defaults.

`cleanup.py` closes external services such as wandb.

`reports.py` owns execution report creation and saving.

## Factory Contract

Trainer configs are plain dictionaries:

```python
{
    "trainer_type": "lightning",
    "max_epochs": 1,
    "accelerator": "auto",
    "devices": "auto",
    "enable_checkpointing": True,
    "enable_progress_bar": True,
    "log_every_n_steps": 1,
}
```

`trainer_type` selects the registered Trainer implementation.

Execution-level logger and callback wiring is not placed inside the Trainer config.

The following keys are rejected in `trainer_config`:

```text
logger
callbacks
```

Use execution-level `loggers` and `checkpoints` configs instead.

## Training Object Contract

`build_training_objects(...)` returns:

```python
{
    "trainer": trainer,
    "module": module,
    "datamodule": datamodule,
}
```

The objects are already fully built and ready for:

```python
trainer.fit(model=module, datamodule=datamodule, ckpt_path=...)
```

## Validation Object Contract

`build_validation_objects(...)` reuses the training object construction path, then applies module loading if configured.

This keeps validation consistent with training while allowing evaluation weights to be restored into an already-built module.

## Resume Contract

Training resume is handled in `train.py` through:

```python
trainer.fit(..., ckpt_path=...)
```

This is separate from `Models/Loading`, which restores weights into an already-built module for validation or evaluation.

## Wandb Contract

Execution applies wandb model watching after both the module and trainer loggers exist.

This belongs in Execution because it needs both objects:

```text
module + trainer loggers + logger config
```

Setup may prepare wandb mode/login, and Loggers may build wandb loggers, but Execution decides when to attach watching to the actual module.

## Subsystem Contract

Execution may:

* orchestrate training and validation runs;
* build object bundles through owning subsystem factories;
* construct Lightning Trainer objects;
* attach logger callbacks and checkpoint callbacks;
* pass resume checkpoint paths to `trainer.fit`;
* call external-service cleanup;
* write execution reports.

Execution must not:

* build datasets or collators directly;
* instantiate model components directly;
* compute rollouts, metrics, or losses;
* implement checkpoint callback logic;
* implement logger internals;
* resolve low-level runtime paths outside Setup-owned context creation.

## Registration Contract

Trainer construction is registered with the decorator-based registry API.

The expected pattern is:

1. define a Trainer adapter class;
2. add its plain default config in `configs.py`;
3. register it with `TRAINER_REGISTRY.register_class(...)`;
4. expose construction through `factory.py`.

Do not add manual Trainer builder registries in `factory.py`.

## Extension Steps

To add a new execution mode:

1. add a mode-specific entry point, such as `test.py` or `predict.py`;
2. reuse object construction from `factory.py` when possible;
3. keep mode-specific orchestration in the entry point;
4. update execution reports if new result fields are needed;
5. add focused smoke tests for the new execution mode;
6. update this README if the execution contract changes.

