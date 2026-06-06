# LightningModules maintenance

## 0. Philosophy

The `LightningModules/` subsystem is responsible for orchestration only.

Lightning modules:
- orchestrate training/validation/test loops,
- call already-built models,
- compute metrics,
- compute losses,
- log outputs,
- configure optimization processes.

They should NOT:
- implement low-level model architectures,
- define metrics,
- define losses,
- contain large amounts of data-processing logic,
- rebuild objects from raw configs internally.

The design emphasizes:
- minimal abstractions,
- clear responsibility separation,
- cooperative multiple inheritance,
- reusable orchestration blocks,
- serializable configurations,
- avoiding duplicated computations.


---

## 1. Folder structure

```text
LightningModules/
- base.py
- tse_base.py
- waveUnet.py
- factory.py
- configs.py
- maintenance.md
```

Additional files may be added later for:

- GAN modules,
- diffusion modules,
- multi-model systems,
- custom optimization schemes,
- human feedback

### base.py

Shared Lightning base classes.

Currently contains:

- BaseLightningModule
- SingleModelOptimizerMixin

Responsibilities:

- define generic Lightning logging helpers;
- define ML-step validation utilities;
- define flat log-dict manipulation helpers;
- define optimizer/scheduler orchestration for single-model systems.

Current logging conventions:
- metric/loss dicts are flat and unprefixed;
- ML prefixes (`train/`, `val/`, `test/`) are added only inside Lightning modules.

Current optimizer conventions:
- optimizers are configured from serializable config dictionaries;
- optimizer instantiation happens in `configure_optimizers`;
- only trainable parameters (`requires_grad=True`) are optimized.

`SingleModelOptimizerMixin` is a mixin:
- it does NOT inherit from `pl.LightningModule`;
- it MUST use cooperative initialization with:
    - `super().__init__(**kwargs)`

Important:
- `super().__init__(**kwargs)` must be called BEFORE assigning any `nn.Module`;
- otherwise PyTorch raises:
    - `cannot assign module before Module.__init__() call`

This file should contain:
- generic Lightning utilities;
- generic optimization helpers for reusable mixins.

This file should NOT contain:
- TSE-specific logic;
- model-specific train/val/test steps;
- metric definitions;
- loss definitions;
- batch parsing assumptions.

### tse_base.py

TSE-specific Lightning base classes.

Currently contains:

- TSEBaseLightningModule
- TSESingleModelLightningModule
- DefaultTSESingleModelLightningModule

Responsibilities:

- define TSE metric/loss orchestration;
- compute metrics from model predictions;
- compute losses from metric dictionaries;
- centralize TSE logging behavior;
- define default TSE train/val/test workflows.

Current TSE conventions:
- TSE metric sets use the interface:
    - `(preds, target, mixture=None, clue=None)`
- metric dictionaries are flat and unprefixed;
- metrics are computed once and reused for:
    - logging;
    - loss computation.

`TSEBaseLightningModule`:
- stores train/val/test metric sets;
- stores the metric-based loss;
- defines:
    - `process_tse_step_outputs`

This class does NOT:
- parse batches;
- call the model;
- define optimizer logic.

`TSESingleModelLightningModule`:
- combines:
    - `SingleModelOptimizerMixin`
    - `TSEBaseLightningModule`
- assumes:
    - one model;
    - one optimizer process.

`DefaultTSESingleModelLightningModule`:
- defines the default/simple TSE workflow.

Current assumptions:
- batch structure:
    - `(mixture, target)`
- model call:
    - `preds = self(mixture)`
- no clue signal.

This file should contain:
- reusable TSE orchestration logic;
- reusable TSE training-step logic;
- reusable single-model TSE workflows.

This file should NOT contain:
- model architecture definitions;
- metric implementations;
- loss implementations;
- config-building logic.

### factory.py

Lightning module construction logic.

Responsibilities:

- map Lightning module names to Lightning module classes;
- build Lightning modules from config dictionaries;
- orchestrate high-level object construction.

This factory may call:
- `Models.factory`
- `Metrics.factory`

Current construction pipeline:
- build models;
- build metric sets;
- build losses;
- gather optimization configs;
- instantiate the requested Lightning module.

Current conventions:
- Lightning module classes receive already-built objects;
- factories receive plain serializable config dictionaries;
- factories are model-count agnostic;
- factories are optimizer-count agnostic.

Current config conventions:
- model configs use:
    - `model_configs`
- optimizer configs use:
    - `optimizer_configs`
- scheduler configs use:
    - `scheduler_configs`

Current single-model modules use:
- `"model"` as the default model key.

Example:

```python
{
    "module_type": "waveunet",

    "model_configs": {
        "model": {...},
    },

    "optimizer_configs": {
        "model": {...},
    },
}
```

This design is intentionally compatible with future:

- GANs;
- teacher/student systems;
- multi-model systems;
- multiple optimizers.

This file should contain:

- Lightning module registries;
- Lightning module construction logic;
- high-level orchestration logic.

This file should NOT contain:

- model definitions;
- metric definitions;
- optimizer implementations;
- training logic.

### configs.py

Default Lightning module configurations.

Responsibilities:

- store default Lightning module config dictionaries;
- assemble reusable end-to-end training configs;
- centralize default module-level experiment settings.

This file gathers defaults from:
- `Models.configs`
- `Metrics.configs`
- `Optimization.configs`

Current configs:
- `DEFAULT_WAVEUNET_LIGHTNING_MODULE_CONFIG`

Current config structure:

```python
{
    "module_type": ...,

    "model_configs": {...},

    "train_metrics_config": ...,
    "val_metrics_config": ...,
    "test_metrics_config": ...,

    "loss_weights": ...,

    "optimizer_configs": {...},
    "scheduler_configs": {...},

    "log_loss_ml_steps": (...),
}
```

Current single-model convention:

- "model" is the default model key.

Example:

"model_configs": {
    "model": DEFAULT_WAVEUNET_CONFIG,
}

This structure is intentionally compatible with future:

- multi-model systems;
- GANs;
- multiple optimizers;
- multiple schedulers.

Experiment-specific modifications should be done outside LightningModules/,
by copying and modifying these defaults.


###  waveUnet.py

WaveUNet Lightning module wrappers.

Currently contains:

- WaveUNetLightningModule

Responsibilities:

- provide a Lightning wrapper around a pre-built `WaveUNet`;
- expose the WaveUNet module type to the Lightning factory;
- optionally specialize WaveUNet-specific behavior later if needed.


## 2. Add a module

Adding a new Lightning module should only modify the files involved in the module pipeline.

### Step 1 : Create the module wrapper

Add a new file if needed:

```text
Modules/
- myModule.py
```

Define the Lightning wrapper class.

Prefer reusing existing base classes and mixins.

### Step 2 : Register the module in factory.py

Import the module class in factory.py.

Add it to the default registry:

"module_name": ModuleClass

The module name should be lowercase and consistent with the project.

### Step 3 : Add default configs if needed

Add a default config in configs.py.

Prefer reusing:

- model configs;
- metric configs;
- optimization configs.

### Step 4 : Check cooperative initialization

All parent classes and mixins must use:

super().__init__(**kwargs)

before assigning nn.Module objects.

### Step 5 : Check factory compatibility

Verify that:

model keys;
optimizer keys;
scheduler keys

match the module constructor expectations.

### Step 6 : Update dependencies if needed

If the module requires new dependencies, add them to the appropriate requirements file.


## 3. Delete a module

Deleting a Lightning module means removing it from every place where it is exposed.

### Step 1 : Remove it from configs.py

Remove its default config if it exists.

### Step 2 : Remove it from factory.py

Remove the module from the default registry.

Also remove its import if unused.

### Step 3 : Delete the wrapper file if unused

Delete the module-specific file only if no other code imports it.

### Step 4 : Check experiment configs

Verify that no experiment config still uses its `module_type`.

### Step 5 : Check imports

Typical places to check:

- factory.py
- configs.py
- notebooks
- experiment scripts