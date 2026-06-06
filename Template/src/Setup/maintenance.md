# Setup Maintenance

This file explains how to cleanly maintain the `Setup/` folder.



# 1. Folder role


`Setup/` prepares the runtime environment before object construction and execution.

Global pipeline:

```text
user config
→ Config/
→ Setup/
→ runtime_context
→ Factory/
→ built objects
→ execution
```

`Setup/` is responsible for:

- environment checks;
- hardware checks;
- reproducibility setup;
- runtime path preparation;
- external logger registration;
- runtime diagnostics.

`Setup/` is NOT responsible for:

- loading configs;
- merging configs;
- building models;
- building metrics;
- building optimizers;
- building Lightning modules;
- launching training/testing/prediction.

# 2. Runtime context philosophy


`Setup/` returns a `runtime_context`.

The runtime context contains resolved runtime facts, for example:

```python
runtime_context = {
    "environment": ...,
    "hardware": ...,
    "reproducibility": ...,
    "paths": ...,
    "logger_registration": ...,
}
```

The runtime context should:

- stay serializable;
- stay explicit;
- avoid hidden mutable state;
- avoid storing secrets/API keys;
- avoid storing large objects unnecessarily.

Factories may consume both:

```text
subsystem configs + runtime_context
```

Example:

```python
runtime_context["paths"]["checkpoints_dir"]
```



# 3. File roles


```text
Setup/

- configs.py
    Default setup configuration dictionaries.

- environment.py
    Runtime import/environment validation.

- hardware.py
    Hardware diagnostics and validation.

- reproducibility.py
    Seed fixing and deterministic runtime setup.

- paths.py
    Runtime output path preparation.

- logger_registration.py
    External logger backend registration/setup.

- user_credential.py
    User-local credential loading and environment variable injection.

- setup.py
    High-level runtime setup orchestration.

- data.py
    Data-related runtime path validation.

- maintenance.md
    Maintenance rules for the Setup subsystem.
```


## configs.py

Stores default setup configuration dictionaries.

It defines defaults for:
- environment checks;
- hardware checks;
- reproducibility;
- runtime paths;
- data paths;
- logger registration.

## environment.py

Checks that required Python imports are available.

It does not install packages.

## hardware.py

Collects hardware information and validates requested hardware constraints.

It does not configure the Lightning Trainer.

## reproducibility.py

Applies seed fixing and backend reproducibility settings.

It handles runtime side effects such as PyTorch deterministic flags.

## paths.py

Resolves runtime paths and prepares run output directories.

It also owns generic path utilities used by other setup files.

## user_credential.py

Loads private, user-local credential files and exports selected values as environment variables.

It is used for credentials such as API keys that should not be stored in public project configs.

It should:
- support config loading through `Configs.conversion.load_config`;
- inject selected values into `os.environ`;
- return only metadata in the runtime context;
- never store secret values in `runtime_context`.

The local credential file should be excluded from version control, for example:

```gitignore
**/user_credential_local.*
```

## logger_registration.py

Registers external logger backends when needed.

Currently handles WandB login/registration.

It does not build Lightning logger objects.

## setup.py

High-level setup orchestrator.

It calls each setup substep and gathers the returned contexts into `runtime_context`.

## data.py

Resolves and validates named dataset root paths.

It does not build datasets, datamodules, splits, or preprocessing pipelines.

## maintenance.md

Documents Setup folder responsibilities and maintenance rules.



# 4. Paths subsystem rules


`paths.py` prepares runtime output directories.

It does not:

- load configs;
- save artifacts;
- understand dataset internals.

Standard structure:

```text
runs/
  experiment_name/

    single_runs/
      run_name/

    sweeps/
      sweep_name/
        run_name/
```

Each run directory contains:

```text
checkpoints/
logs/
metrics/
configs/
artifacts/
```



# 6. Logger registration rules


`logger_registration.py` prepares external logger backends.

It does not build Lightning loggers.

Current implementation:

- WandB registration/login.

Future logger backends should follow the dispatcher pattern:

```python
LOGGER_REGISTRATION_DISPATCHER = {
    ...
}
```

Each backend should:

- expose one setup function;
- return a serializable context;
- avoid storing secrets.

Logger registration should not load private credential files directly.

If a logger needs an API key, it should read it from an environment variable.  
The environment variable may be set externally or injected earlier by `user_credential.py`.

# 7. Adding a new setup subsystem


Example: adding distributed runtime checks.

Steps:

1. Create a dedicated file:
   ```text
   distributed.py
   ```

2. Add:
   - one public setup entry point;
   - helper functions;
   - returned runtime context.

3. Add default config in:
   ```python
   DEFAULT_SETUP_CONFIG
   ```

4. Register the setup step inside:
   ```python
   setup_runtime()
   ```

5. Update this maintenance file.


