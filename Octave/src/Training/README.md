# Optimization README

This folder owns optimizer construction for Octave.

Optimizers are delayed-construction objects:

```text
optimizer_config -> optimizer_builder
optimizer_builder + parameters -> torch optimizer
```

The optimizer builder can be created before model parameters are available. The actual PyTorch optimizer is created later, once parameters are supplied.

## File Roles

```text
Optimization/
- configs.py
- registry.py
- optimizers.py
- factory.py
- README.md
```

`configs.py` stores plain optimizer config dictionaries.

`registry.py` defines:

```text
OPTIMIZER_REGISTRY
OPTIMIZER_BUILDER
```

`optimizers.py` defines registered optimizer-builder classes.

Each registered class maps one optimizer type to one PyTorch optimizer class.

`factory.py` exposes:

```text
build_optimizer
build_optimizer_builder
```

It imports optimizer implementation files so decorator registration is executed, then delegates optimizer-builder construction to `src/Workflow/Factory`.

## Factory Contract

Optimizer configs are plain dictionaries:

```python
{
    "optimizer_type": "adamw",
    "lr": 0.001,
    "betas": [0.9, 0.999],
    "eps": 1e-8,
    "weight_decay": 0.01,
}
```

`optimizer_type` selects the registered optimizer-builder class.

The optimizer factory must not implement manual optimizer registries or manual default merging. Generic registry lookup, config validation, default merging, routing-field removal, and object construction belong to `src/Workflow/Factory`.

## Delayed Construction Contract

`build_optimizer_builder(...)` returns a callable object.

The returned object is later called with model parameters:

```python
optimizer_builder(parameters)
```

It returns a PyTorch optimizer.

This keeps parameter ownership outside the factory. The model or Lightning module decides which parameters are optimized; this subsystem only turns optimizer configs into optimizer construction policies.

## Registered Optimizer Types

`adamw`

* builds `torch.optim.AdamW`;
* supports the AdamW default config from `configs.py`.

`adam`

* builds `torch.optim.Adam`;
* supports the Adam default config from `configs.py`.

`sgd`

* builds `torch.optim.SGD`;
* supports the SGD default config from `configs.py`.

## Subsystem Contract

Optimization may:

* parse optimizer configs;
* create delayed callable optimizer builders;
* instantiate PyTorch optimizers once parameters are supplied;
* adapt YAML-friendly values such as `betas: [0.9, 0.999]` into PyTorch-friendly tuples.

Optimization must not:

* decide which model components exist;
* decide which parameters are trainable;
* build models;
* build schedulers;
* run training;
* read run configs directly.

## Registration Contract

Optimizers are registered with the decorator-based registry API.

The expected pattern is:

1. define an optimizer-builder class;
2. set its `optimizer_class`;
3. add its plain default config in `configs.py`;
4. register the class with `OPTIMIZER_REGISTRY.register_class(...)`;
5. expose construction through `factory.py`.

Do not add manual `OPTIMIZER_REGISTRY = {...}` dictionaries in `factory.py`.

## Validation Responsibility

Generic validation belongs to `src/Workflow/Factory`, including:

* selecting the registered optimizer-builder class from `optimizer_type`;
* checking config keys against the registered default config;
* merging defaults;
* removing the routing field before constructor call;
* instantiating the configured optimizer-builder object.

Optimizer-specific validation belongs here, including:

* preserving delayed construction until parameters exist;
* adapting YAML-friendly optimizer values to PyTorch constructor values.

## Extension Steps

To add an optimizer:

1. add a plain default config in `configs.py`;
2. create an optimizer-builder class in `optimizers.py`;
3. set its `optimizer_class`;
4. register it with `OPTIMIZER_REGISTRY.register_class(...)`;
5. import the implementation module in `factory.py` so registration is executed;
6. add focused tests for registration, config validation, delayed construction, and PyTorch optimizer construction;
7. update this README if the subsystem contract changes.

