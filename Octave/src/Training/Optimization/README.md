# Optimization README

This folder owns optimizer construction for Octave.

## File Roles

`configs.py`

- stores plain optimizer config dictionaries.

`factory.py`

- builds PyTorch optimizers from plain dictionaries;
- builds configured optimizer builder objects for Lightning modules;
- validates optimizer type and config keys.

## Subsystem Contract

Optimization configs stay plain and serializable.

This subsystem may:

- map optimizer config names to PyTorch optimizer classes;
- attach parameters supplied by a caller;
- expose configured callable builders for modules whose parameters are only
  available inside `configure_optimizers`;
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

`build_optimizer_builder` returns an already-configured callable object. A
Lightning module may call that object with `self.parameters()`, but it does not
own config parsing.

## Extension Steps

1. Add default config values in `configs.py`.
2. Register the optimizer class in `factory.py`.
3. Add focused unit tests.
4. Update this README when optimizer or scheduler contracts change.
