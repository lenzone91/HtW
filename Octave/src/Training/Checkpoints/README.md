# Checkpoints README

This folder owns Lightning checkpoint callback construction.

## File Roles

`configs.py`

- stores plain checkpoint callback config dictionaries.

`factory.py`

- builds Lightning `ModelCheckpoint` callbacks;
- resolves checkpoint directories through `runtime_context`.

## Subsystem Contract

Checkpoints may:

- build checkpoint callbacks from plain configs;
- resolve callback `dirpath`;
- support last, periodic, and monitored best-value checkpoints.

Checkpoints must not:

- run training;
- resume trainer state;
- load model weights.

## Resume Contract

Training resume belongs to `Execution/train.py` via:

```python
trainer.fit(..., ckpt_path=...)
```

Checkpoint callbacks save state. Execution decides whether to resume from one.

## Loading Contract

Validation/evaluation loading belongs to `Models/Loading`. Checkpoints only
create the files and callbacks that make those weights available.
