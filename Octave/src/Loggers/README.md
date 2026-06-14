# Loggers README

This folder owns Lightning logger construction for Octave.

Loggers are built from plain logger configs:

```text
logger_configs -> Lightning logger objects | False
```

An empty logger config disables Lightning logging.

## File Roles

```text
Loggers/
- configs.py
- registry.py
- loggers.py
- factory.py
- wandb_metrics.py
- README.md
```

`configs.py` stores plain logger config dictionaries.

`registry.py` defines:

```text
LOGGER_REGISTRY
LOGGER_BUILDER
```

`loggers.py` defines registered logger classes.

It registers CSV and wandb logger adapters through the decorator-based factory API and owns logger-specific path resolution.

`factory.py` exposes:

```text
build_loggers
build_logger
build_logger_callbacks
get_wandb_metrics_config
get_wandb_watch_config
watch_module_with_wandb_loggers
normalize_loggers
```

It imports logger implementation files so decorator registration is executed, handles wandb watch and metrics helper configs, and delegates logger object construction to `src/Workflow/Factory`.

`wandb_metrics.py` defines the wandb scalar metrics callback.

It forwards scalar Lightning metrics to wandb history and fails early if wandb metrics are configured but no `WandbLogger` is attached.

## Factory Contract

Empty logger config disables Lightning logging:

```python
{}
```

Named logger configs build one logger per outer key:

```python
{
    "csv": {
        "logger_type": "csv",
        "save_dir": "logs",
        "name": "csv",
        "version": None,
        "prefix": "",
        "flush_logs_every_n_steps": 100,
    },
    "wandb": {
        "logger_type": "wandb",
        "project": "htw-ac-video-jepa",
        "name": "toy",
        "offline": True,
        "watch": {
            "enabled": True,
            "log": "all",
            "log_freq": 100,
            "log_graph": True,
        },
        "metrics": {
            "enabled": True,
            "define_metrics": True,
            "direct_log": True,
            "step_metric": "trainer/global_step",
            "metric_patterns": ["train/*", "val/*", "test/*"],
            "require_wandb_logger": True,
        },
    },
}
```

The outer config key is the logger instance name.

The inner `logger_type` selects the registered logger implementation.

The logger factory must not implement manual logger registries or manual default merging. Generic registry lookup, config validation, default merging, field resolution, and object construction belong to `src/Workflow/Factory`.

## Runtime Context Contract

Logger path resolution uses `runtime_context`.

Relative logger paths are resolved from:

```text
runtime_context["paths"]["run_dir"]
```

Absolute paths are resolved directly.

When `runtime_context` is `None`, relative paths remain relative.

This resolution belongs to the logger subsystem because loggers own where their files and dashboard artifacts are written.

## Wandb Scope

Wandb is supported for metric logging and dashboards.

Wandb sweeps are intentionally out of scope.

Wandb parameter and gradient watching is configured through the logger config, then applied by execution after the Lightning module has been built.

Wandb scalar metrics can also be backed by a callback. This protects against runs where W&B system metrics and terminal logs appear, but Lightning scalar metrics do not show up as W&B history keys or charts.

## Subsystem Contract

Loggers may:

* construct Lightning loggers from plain configs;
* resolve logger output directories;
* configure wandb parameter and gradient watching;
* build wandb scalar metric callbacks;
* return `False` when logging is disabled.

Loggers must not:

* run training;
* own model metrics or losses;
* perform sweeps;
* mutate run configs;
* construct models, datamodules, or trainers.

## Registration Contract

Loggers are registered with the decorator-based registry API.

The expected pattern is:

1. define a logger adapter class;
2. add its plain default config in `configs.py`;
3. register the class with `LOGGER_REGISTRY.register_class(...)`;
4. add field resolutions if constructor adaptation is required;
5. expose construction through `factory.py`.

Do not add manual `LOGGER_REGISTRY = {...}` dictionaries in `factory.py`.

## Extension Steps

To add a logger:

1. add a plain default config in `configs.py`;
2. create a logger adapter class in `loggers.py`;
3. register it with `LOGGER_REGISTRY.register_class(...)`;
4. add field resolutions for path or constructor adaptation if required;
5. import the implementation module in `factory.py` so registration is executed;
6. add focused tests for registration, config validation, path resolution, disabled logging, and logger construction;
7. update this README if the subsystem contract changes.

