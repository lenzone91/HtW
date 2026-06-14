# Schedulers README

This folder owns learning-rate scheduler construction for Octave.

## File Roles

`configs.py`

- stores plain scheduler config dictionaries.

`factory.py`

- builds PyTorch schedulers from plain dictionaries;
- builds configured scheduler builder objects for Lightning modules;
- returns Lightning-compatible scheduler dictionaries;
- validates scheduler type and config keys.

## Subsystem Contract

Schedulers may:

- map scheduler config names to PyTorch scheduler classes;
- attach schedulers to an optimizer supplied by a caller;
- expose configured callable builders for modules whose optimizer is only
  available inside `configure_optimizers`;
- provide Lightning metadata such as `interval`, `frequency`, and `monitor`;
- reject unknown scheduler keys early.

Schedulers must not:

- build optimizers;
- decide which model parameters are trained;
- run training;
- read run configs directly.

## Factory Contract

Disabled scheduler config:

```python
{
    "enabled": False,
}
```

Enabled scheduler config:

```python
{
    "enabled": True,
    "scheduler_type": "step_lr",
    "step_size": 1,
    "gamma": 0.1,
    "interval": "epoch",
    "frequency": 1,
}
```

Supported scheduler types are `step_lr`, `cosine_annealing_lr`, and
`reduce_lr_on_plateau`.

`build_scheduler_builder` returns an already-configured callable object. A
Lightning module may call that object with the optimizer returned by its
optimizer builder, but it does not own config parsing.
