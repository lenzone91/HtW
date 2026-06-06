# Optimization Maintenance

This file explains how to maintain the `Optimization/` folder.

## 1. File roles

```text
Optimization/
- configs.py
- factory.py
- maintenance.md
```

### configs.py

Stores reusable optimizer and scheduler configs.

Configs should be plain serializable dictionaries.

### factory.py

Maps config names to PyTorch optimizer/scheduler classes.

Responsibilities:

- define optimizer and scheduler registries;
- build optimizers from model parameters and config;
- build schedulers from optimizers and config.

Instantiation will usually be called inside a custom LightningModule,
typically in configure_optimizers. 

## 2. Adding an optimizer

1. Add the optimizer class to DEFAULT_OPTIMIZER_CLASSES.
2. Add or update a default config in configs.py if useful.
3. Keep optimizer-specific kwargs inside the config dictionary.

## 3. Adding a scheduler

1. Add the scheduler class to DEFAULT_SCHEDULER_CLASSES.
2. Add or update a default config in configs.py if useful.
3. If the scheduler requires Lightning-specific metadata, such as monitor,
handle that in the LightningModule return format, not inside the scheduler object.


## 4. Design rules

- Do not put experiment-specific configs here.
- Do not instantiate optimizers globally.
- Do not make this folder depend on TSE logic.
- Lightning still receives standard PyTorch optimizer/scheduler objects.