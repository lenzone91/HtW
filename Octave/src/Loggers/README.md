# Loggers README

This folder owns Lightning logger construction for Octave.

## File Roles

`configs.py`

- stores plain logger config dictionaries.

`factory.py`

- builds Lightning loggers from plain configs;
- supports CSV and wandb loggers;
- configures wandb parameter and gradient watching;
- builds wandb scalar metric callbacks;
- resolves logger paths through `runtime_context`.

`wandb_metrics.py`

- defines wandb metric patterns;
- forwards scalar Lightning metrics to wandb history;
- fails early if wandb metrics are configured but no `WandbLogger` is attached.

## Subsystem Contract

Loggers may:

- construct dashboard and file loggers;
- make wandb scalar metric history explicit;
- resolve logger output directories;
- return `False` when logging is disabled.

Loggers must not:

- run training;
- own metrics;
- perform sweeps;
- mutate run configs.

## Wandb Scope

Wandb is supported for metric logging and dashboards.

Wandb sweeps are intentionally out of scope.

Wandb parameter and gradient watching is configured through the logger config,
then applied by execution after the Lightning module has been built.

Wandb scalar metrics are also backed by a callback. This protects against runs
where W&B system metrics and terminal logs appear, but Lightning scalar metrics
do not show up as W&B history keys or charts.

## Factory Contract

Empty logger config disables Lightning logging:

```python
{}
```

Wandb logging uses:

```python
{
    "wandb": {
        "project": "htw-ac-video-jepa",
        "name": "toy",
        "offline": true,
        "watch": {
            "enabled": true,
            "log": "all",
            "log_freq": 100,
            "log_graph": true,
        },
        "metrics": {
            "enabled": true,
            "define_metrics": true,
            "direct_log": true,
            "step_metric": "trainer/global_step",
            "metric_patterns": ["train/*", "val/*", "test/*"],
            "require_wandb_logger": true,
        },
    }
}
```
