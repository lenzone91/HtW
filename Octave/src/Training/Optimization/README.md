# Optimization README

This folder owns optimizer construction for Octave.

## File Roles

`configs.py`

- stores plain optimizer config dictionaries.

`factory.py`

- builds PyTorch optimizers from plain dictionaries;
- validates optimizer type and config keys.

## Subsystem Contract

Optimization configs stay plain and serializable.

This subsystem may:

- map optimizer config names to PyTorch optimizer classes;
- attach parameters supplied by a caller;
- reject unknown optimizer keys early.

This subsystem must not:

- decide which model components exist;
- build models;
- run training;
- read run configs directly.

## Factory Contract

Optimizer configs use:

```python
{
    "optimizer_type": "adamw",
    "lr": 0.001,
    ...
}
```

Schedulers are owned by `../Schedulers`.

## Extension Steps

1. Add default config values in `configs.py`.
2. Register the optimizer class in `factory.py`.
3. Add focused unit tests.
4. Update this README when optimizer or scheduler contracts change.
