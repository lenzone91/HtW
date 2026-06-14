# Schedulers README

This folder owns learning-rate scheduler construction for Octave.

Schedulers are delayed-construction objects:

```text
scheduler_config -> scheduler_builder
scheduler_builder + optimizer -> Lightning scheduler dictionary
```

The scheduler builder can be created before the optimizer exists. The actual PyTorch scheduler is created later, once an optimizer is available.

## File Roles

```text
Schedulers/
- configs.py
- registry.py
- schedulers.py
- factory.py
- README.md
```

`configs.py` stores plain scheduler config dictionaries.

`registry.py` defines:

```text
SCHEDULER_REGISTRY
SCHEDULER_BUILDER
```

`schedulers.py` defines registered scheduler-builder classes.

Each registered class maps one scheduler type to one PyTorch scheduler class.

`factory.py` exposes:

```text
build_scheduler
build_scheduler_builder
is_scheduler_enabled
```

It imports scheduler implementation files so decorator registration is executed, handles disabled scheduler configs, and delegates enabled scheduler construction to `src/Workflow/Factory`.

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
    "last_epoch": -1,
    "interval": "epoch",
    "frequency": 1,
    "monitor": None,
    "strict": True,
    "name": None,
}
```

`scheduler_type` selects the registered scheduler-builder class.

The scheduler factory must not implement manual scheduler registries or manual default merging. Generic registry lookup, config validation, default merging, and object construction belong to `src/Workflow/Factory`.

## Delayed Construction Contract

`build_scheduler_builder(...)` returns a callable object.

For enabled scheduler configs, the returned object is a configured scheduler builder:

```python
scheduler_builder(optimizer)
```

It returns a Lightning-compatible scheduler dictionary:

```python
{
    "scheduler": scheduler,
    "interval": "epoch",
    "frequency": 1,
    ...
}
```

For disabled scheduler configs, the returned object returns `None` when called.

## Registration Contract

Schedulers are registered with the decorator-based registry API.

The expected pattern is:

1. define a scheduler-builder class;
2. set its `scheduler_class`;
3. add its plain default config in `configs.py`;
4. register the class with `SCHEDULER_REGISTRY.register_class(...)`;
5. expose construction through `factory.py`.

Do not add manual `SCHEDULER_REGISTRY = {...}` dictionaries in `factory.py`.

## Subsystem Contract

Schedulers may:

* parse scheduler configs;
* create delayed callable scheduler builders;
* instantiate PyTorch schedulers once an optimizer is supplied;
* return Lightning-compatible scheduler dictionaries;
* expose Lightning metadata such as `interval`, `frequency`, `monitor`, `strict`, and `name`.

Schedulers must not:

* build optimizers;
* decide which model parameters are trained;
* build models or datamodules;
* run training;
* resolve experiment paths;
* read run configs directly.

## Validation Responsibility

Generic validation belongs to `src/Workflow/Factory`, including:

* selecting the registered scheduler-builder class from `scheduler_type`;
* checking config keys against the registered default config;
* merging defaults;
* instantiating the configured scheduler-builder object.

Scheduler-specific validation belongs here, including:

* detecting whether a scheduler is enabled;
* preserving delayed construction until an optimizer exists;
* separating Lightning scheduler metadata from PyTorch scheduler kwargs.

## Extension Steps

To add a scheduler:

1. add a plain default config in `configs.py`;
2. create a scheduler-builder class in `schedulers.py`;
3. set its `scheduler_class`;
4. register it with `SCHEDULER_REGISTRY.register_class(...)`;
5. import the implementation module in `factory.py` so registration is executed;
6. add focused tests for registration, disabled behavior, config validation, and delayed construction;
7. update this README if the subsystem contract changes.

