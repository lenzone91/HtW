# Checkpoints Maintenance

This file explains how to cleanly maintain the `Checkpoints/` folder.


## 1. File roles

The `Checkpoints/` folder is organized by responsibility.

```text
Checkpoints/
- factory.py
- configs.py
- maintenance.md
```

### factory.py

Checkpoint callback construction logic.

Responsibilities:

- map checkpoint types to callback builders;
- instantiate Lightning checkpoint callbacks from config dictionaries;
- centralize all checkpoint-building logic.

Currently supported checkpoint types:

- last checkpoint;
- periodic checkpoint;
- best metric-value checkpoint.

Checkpoint callbacks should operate on logged scalar names only.

They should not depend on:

- metric classes;
- TSEMetricSet internals;
- loss internals.

Losses and metrics are treated identically at checkpoint level.

Examples:

- val/loss
- val/sisdr
- val/pesq

The monitored quantity is simply a logged scalar key.


### configs.py

Default checkpoint configurations.

Responsibilities:

- store reusable checkpoint config dictionaries;
- provide default checkpoint templates;
- define default monitoring conventions.

### maintenance.md

Checkpoint maintenance protocol.

Responsibilities:

- explain how to add checkpoint types;
- explain how to delete checkpoint types;
- document checkpoint naming and monitoring conventions.


## 2. Adding a checkpoint type

Adding a checkpoint type should update only the files that are relevant to the checkpoint pipeline.

### Step 1 : Define the checkpoint behavior

First determine the checkpoint trigger logic.

Current checkpoint families:

- periodic saving;
- best metric-value saving;
- last checkpoint saving.

A checkpoint callback should operate on:
- training state;
- logged scalar values;
- checkpoint frequency.

It should not depend on:
- metric implementations;
- loss implementations;
- TSE-specific logic.

### Step 2 : Decide whether Lightning already supports it

Before implementing a custom callback, check whether the behavior is already supported by:

```python
lightning.pytorch.callbacks.ModelCheckpoint
```

If the behavior already exists in Lightning, prefer wrapping/configuring the native callback instead of subclassing it.

Prefer minimal wrappers around Lightning callbacks.

### Step 3 : Add the builder in factory.py

Create a dedicated builder function.

The builder should:

- receive a plain config dictionary;
- return a Lightning callback object;
- avoid hidden side effects.

### Step 4 : Register the checkpoint type

Add the builder to:

```python
CHECKPOINT_CALLBACK_BUILDERS
```

Format:

```python
"checkpoint_type": builder_function
```


Checkpoint type names should stay:
- lowercase;
- short;
- behavior-oriented.

### Step 5 : Add a default config if needed

If the checkpoint type is expected to be reused frequently, add a default config dictionary in configs.py and update the overall default configs dictionnary.


### Step 6 : Respect monitoring conventions

Checkpoint monitoring should use logged scalar keys exactly as they appear in Lightning logs.


Checkpoint code should not reconstruct prefixes manually.

Checkpoint code should not know whether a monitored quantity is:
- a loss;
- an objective metric;
- a perceptual metric.

At checkpoint level, all monitored quantities are treated as scalar logged values.

### Step 7 : Prefer minimal checkpoint logic

Avoid implementing:
- experiment-specific logic;
- metric-specific logic;
- dataset-specific logic;
- model-specific logic.

Checkpoint callbacks should remain generic and reusable across experiments.


## 3. Deleting a checkpoint type

Deleting a checkpoint type means removing it from every place where it is exposed to the checkpoint pipeline.

### Step 1 : Remove it from configs.py

If the checkpoint type appears in a default config, remove its entry.

Also remove its default config dictionary if it is no longer used.

### Step 2 : Remove it from factory.py

Remove the checkpoint type from:

```python
CHECKPOINT_CALLBACK_BUILDERS
```

Also remove its builder function if no other code uses it.

### Step 3 : Remove unused imports

If the deleted builder was the only user of an import, remove that import.

Typical import to check:

from lightning.pytorch.callbacks import ModelCheckpoint

Only remove it if no remaining checkpoint builder uses it.

### Step 4 : Check experiment configs

Verify that no experiment config still refers to the deleted checkpoint type.

Typical field to search:

"checkpoint_type": "..."

### Step 5 : Check training entry points

If the checkpoint type was explicitly added in a training script or notebook, remove or replace it there.

Typical places to check:

training scripts;
sanity-check notebooks;
experiment configs.

### Step 6 : Update maintenance.md

Remove references to the deleted checkpoint type from the documentation if it was listed as supported.