# Modules README

This folder owns LightningModule orchestration for AcVideoJepa.

## File Roles

`base.py`

- defines logging helpers shared by Lightning modules;
- forwards explicit Lightning logging options to `self.log_dict`;
- validates ML-step names and log dictionaries.

`ac_video_jepa_module.py`

- defines AcVideoJepa Lightning orchestration;
- receives already-built architecture components, rollout behavior, and objective
  objects;
- receives already-built optimizer and scheduler builder objects;
- parses collated batch dictionaries;
- calls rollout behavior without embedding rollout algorithms;
- calls objective behavior without embedding loss internals;
- logs train losses on step and epoch;
- logs validation and test losses on epoch;
- emits scalar logs to configured Lightning loggers.

`configs.py`

- stores plain default Lightning module configs;
- exposes separate architecture-component, rollout, objective, optimizer, and
  scheduler config sections.

`factory.py`

- builds architecture components through `Model/ac_video_jepa/factory.py`;
- builds rollout behavior through `Rollouts/factory.py`;
- builds objective behavior through `Metrics/factory.py`;
- builds optimizer and scheduler builders through `Training/`;
- passes already-built objects into `AcVideoJepaModule`.

## Config Contract

AcVideoJepa module configs use separate sections:

```python
{
    "components_config": {...},
    "rollout_config": {...},
    "objective_config": {...},
    "optimizer_config": {...},
    "scheduler_config": {...},
}
```

Architecture, rollout, and objective settings should not be mixed.

## Subsystem Contract

Lightning modules may:

- parse collated batch dictionaries;
- register architecture components directly on the Lightning module;
- call already-built objectives;
- call already-built optimizer and scheduler builders inside
  `configure_optimizers`;
- log flat loss dictionaries;

Lightning modules must not:

- build model architectures directly;
- implement rollout algorithms directly;
- instantiate prediction losses or regularizers directly;
- build optimizers or schedulers directly from plain configs;
- build datasets or collators;
- resolve paths;
- depend on hidden global config.

## Batch Contract

The module expects collated batches with:

```python
{
    "states": tensor,
    "actions": tensor,
    ...
}
```

`states` and `actions` are passed to the configured rollout and objective.

## Logging Contract

Training metrics are logged with:

```python
on_step=True
on_epoch=True
logger=True
```

Validation and test metrics are logged with:

```python
on_step=False
on_epoch=True
logger=True
```

This keeps progress-bar feedback and dashboard scalar curves aligned.

## Extension Steps

1. Keep architecture construction in the model factory.
2. Keep batch construction in collators.
3. Keep rollout behavior in `Rollouts/`.
4. Keep objective behavior in `Metrics/`.
5. Add module orchestration in `ac_video_jepa_module.py`.
6. Add focused module tests.
7. Update this README when training-step behavior changes.
