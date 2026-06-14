# Modules README

This folder owns LightningModule orchestration for AcVideoJepa.

A module receives already-built runtime objects and defines Lightning step behavior:

```text
batch -> rollout -> metric_set -> loss -> logs
```

It does not build datasets, collators, model components, metrics, losses, optimizers, or schedulers directly.

## File Roles

```text
Modules/
- base.py
- ac_video_jepa_module.py
- registry.py
- factory.py
- configs.py
- README.md
```

`base.py` defines shared Lightning logging helpers.

`ac_video_jepa_module.py` defines `AcVideoJepaModule`.

It registers the module with `MODULE_REGISTRY` through the decorator-based factory API and owns Lightning step orchestration.

`registry.py` defines:

```text
MODULE_REGISTRY
MODULE_BUILDER
```

`factory.py` exposes:

```text
build_ac_video_jepa_module
```

It builds required dependencies through their owning factories, then delegates final module construction to `src/Workflow/Factory`.

`configs.py` stores plain module configs.

## Factory Contract

The public module config is a plain dictionary:

```python
{
    "module_type": "ac_video_jepa",
    "components_config": {...},
    "rollout_config": {...},
    "metric_set_config": {...},
    "loss_config": {...},
    "optimizer_config": {...},
    "scheduler_config": {...},
    "strict": True,
}
```

The factory builds:

```text
components_config  -> encoder/action_encoder/predictor/encoder_shape
rollout_config     -> rollout
metric_set_config  -> metric_set
loss_config        -> loss
optimizer_config   -> optimizer_builder
scheduler_config   -> scheduler_builder
```

Then it constructs `AcVideoJepaModule`.

## JEPA Split Contract

The migrated module does not receive an objective object.

The old path:

```text
rollout_output -> AcVideoJepaObjective -> loss, logs
```

is obsolete.

The migrated path is:

```text
rollout_output -> metric_set -> metric_values
metric_values  -> loss       -> total_loss, loss_logs
```

This keeps metric computation and scalar loss aggregation separated.

## Step Contract

The module expects collated batches with:

```python
{
    "states": tensor,
    "actions": tensor,
    ...
}
```

Step logic:

1. validate the batch;
2. compute latent rollout from `states` and `actions`;
3. compute flat metric values with `metric_set`;
4. compute scalar loss with `loss`;
5. merge metric logs and loss logs;
6. log through Lightning.

## Logging Contract

Training logs use:

```python
on_step=True
on_epoch=True
logger=True
```

Validation and test logs use:

```python
on_step=False
on_epoch=True
logger=True
```

This preserves terminal feedback and dashboard scalar curves.

## Subsystem Contract

Lightning modules may:

* parse collated batch dictionaries;
* call already-built architecture components;
* call already-built rollout behavior;
* call already-built metric sets;
* call already-built loss modules;
* call already-built optimizer and scheduler builders;
* define Lightning step behavior;
* log flat dictionaries.

Lightning modules must not:

* build datasets or collators;
* instantiate encoders, predictors, metrics, losses, optimizers, or schedulers directly;
* implement rollout algorithms;
* resolve paths;
* read run configs directly;
* depend on the obsolete `AcVideoJepaObjective`.

## Registration Contract

Modules are registered with the decorator-based registry API.

The expected pattern is:

1. define the LightningModule class;
2. add constructor-level default config in `configs.py`;
3. register the class with `MODULE_REGISTRY.register_class(...)`;
4. expose public module construction through `factory.py`.

Do not add manual module builder registries in `factory.py`.

## Extension Steps

To add a module:

1. define the LightningModule class;
2. define its public config and constructor config;
3. register the module class with `MODULE_REGISTRY`;
4. build dependencies through their owning factories;
5. delegate final construction to `src/Workflow/Factory`;
6. add focused tests for dependency construction, batch validation, step output, and optimizer configuration;
7. update this README if step behavior changes.
