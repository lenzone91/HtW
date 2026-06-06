# Sweeps Maintenance

This file explains how to cleanly maintain the `Sweeps/` folder.


## 1. File roles

The `Sweeps/` folder is organized by sweep backend.

```text
Sweeps/
- wandb_sweep.py
- wandb_factory.py
- configs.py
- maintenance.md
```

### wandb_sweep.py

WandB sweep orchestration.

Responsibilities:

- create a new WandB sweep;
- launch a WandB agent;
- attach to an existing WandB sweep;
- initialize one WandB trial;
- call run_training from Execution/train.py.

This file should not mutate project configs directly.

### wandb_factory.py

WandB trial config construction.

Responsibilities:

- copy the base training config;
- inject WandB sampled hyperparameters into the copied config;
- update trial metadata, such as the run name.

This file should not create WandB sweeps, launch agents, or run training.

### configs.py

Default WandB sweep configs.

Responsibilities:

- define the default config for creating a new WandB sweep;
- define the default config for attaching to an existing WandB sweep;
- reuse Execution/ training defaults as the base config.

This file should not redefine training, model, metric, data, logger, or callback defaults.

### maintenance.md

Sweep maintenance protocol.

Responsibilities:

- explain file roles;
- explain how to add a sweep backend;
- explain how to delete a sweep backend;
- document minimal consistency checks after changes.

## 2. Adding a new sweep backend

Adding a sweep backend means adding support for a new tool or algorithm that decides which training configs should be run.

Examples:

- local grid search;
- local random search;
- Optuna;
- Ray Tune;
- custom Bayesian search.
### Step 1 : Add a backend orchestration file

Create one backend-specific file.

Example: optuna_sweep.py

This file should:

- initialize the backend;
- receive or sample hyperparameters;
- build a trial config through a factory helper;
- call run_training.

It should not duplicate the training loop implemented in Execution/train.py.

### Step 2 : Add a backend factory file if needed

If the backend requires config mutation or sampled-parameter injection, add a factory file.

Example: optuna_factory.py

The factory should:

- copy the base config;
- inject sampled parameters;
- set trial metadata.

It should not run training.

### Step 3 : Add default configs

Add the backend default config in configs.py.

Example: DEFAULT_OPTUNA_SWEEP_CONFIG

The config should contain:

- a base execution config;
- backend-specific sweep settings;
- backend-specific runtime settings.

### Step 4 : Keep parameter naming consistent

Sweep parameters should use dotted paths into the base config.

Example:

- trainer.max_epochs
- module.model_configs.model.hidden_channels

This keeps sweep logic independent from the internal structure of model or trainer builders.

## 3. Deleting a sweep backend

Deleting a backend means removing every file and config exposing it.

### Step 1 : Remove the backend orchestration file

Example: optuna_sweep.py
### Step 2 : Remove the backend factory file

Example: optuna_factory.py

Only delete it if no other backend still imports it.

### Step 3 : Remove its default config

Remove the associated config from configs.py.

Example: DEFAULT_OPTUNA_SWEEP_CONFIG
### Step 4 : Clean imports

Check and remove unused imports in:

- configs.py;
- notebooks;
- experiment scripts;
- any optional __init__.py.

### Step 5 : Clean dependencies if needed

If the deleted backend was the only user of an external package, remove that package from the relevant requirements file.

## 4. Modifying WandB sweeps

### Add a new swept parameter

Add it to:

DEFAULT_WANDB_SWEEP_CONFIG["wandb_sweep"]["parameters"]

The key should be a dotted path into:

DEFAULT_WANDB_SWEEP_CONFIG["base_config"]

Example:

"trainer.max_epochs": {
    "values": [5, 10],
}
### Change the optimization target

Update:

DEFAULT_WANDB_SWEEP_CONFIG["wandb_sweep"]["metric"]

Example:

"metric": {
    "name": "val/sisdr",
    "goal": "maximize",
}

The metric name must match the name logged by the Lightning module.

### Use an existing sweep

Use:

DEFAULT_EXISTING_WANDB_SWEEP_CONFIG

This requires:

"sweep_id": ...

and does not create a new sweep.

## 5. Config injection rules

WandB sampled parameters are injected by wandb_factory.py.

Rules:

- the base config is deep-copied before mutation;
- swept keys must already exist in the copied config;
- intermediate keys must point to dictionaries;
- missing keys raise errors.

This prevents silent creation of invalid config fields.

If a parameter must be swept, first make sure it exists in the base execution config.

## 6. Relation with Execution/

Sweeps/ is a meta-execution layer.

It decides which configs should be trained.

Execution/ runs one resolved config.

Expected flow:

Sweeps/
-> build trial config
-> Execution/train.py
-> run_training(trial_config)

Do not move training logic into Sweeps/.

Do not make Execution/train.py aware of sweeps.

## 7. Sanity check of the sweep pipeline

After modifying Sweeps/, check that:

- wandb_sweep.py still calls run_training;
- wandb_factory.py still deep-copies the base config;
- swept parameters are valid dotted paths;
- run names are unique per trial;
- the WandB metric name matches an actually logged metric;
- the base config is a valid Execution/ training config;
- optional WandB imports stay inside WandB-specific files;
- non-WandB execution remains usable without importing WandB.

## Core rule:

If a change is about how one training run is executed, it belongs in Execution/.

If a change is about how many training configs are generated or scheduled, it belongs in Sweeps/.