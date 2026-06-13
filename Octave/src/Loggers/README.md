# Loggers README

This folder owns Lightning logger construction for Octave.

## File Roles

`configs.py`

- stores plain logger config dictionaries.

`factory.py`

- builds Lightning loggers from plain configs;
- supports CSV and wandb loggers;
- configures wandb parameter and gradient watching;
- resolves logger paths through `runtime_context`.

## Subsystem Contract

Loggers may:

- construct dashboard and file loggers;
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
    }
}
```
