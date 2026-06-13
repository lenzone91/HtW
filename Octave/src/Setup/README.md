# Setup README

This folder owns runtime setup for Octave executions.

## File Roles

`setup.py`

- orchestrates setup steps;
- returns `runtime_context`.

`configs.py`

- stores plain setup defaults.

`config_resolution.py`

- loads YAML/JSON configs;
- recursively merges override dictionaries.

`paths.py`

- resolves project and run paths;
- creates run directories.

`reproducibility.py`

- sets random seeds and deterministic flags.

`logger_registration.py`

- prepares external logger runtime state;
- supports wandb login/mode setup.

## Subsystem Contract

Setup may:

- load plain config files;
- merge config overrides;
- resolve runtime paths;
- create runtime directories;
- seed runtime libraries;
- prepare wandb runtime mode/login.

Setup must not:

- build datasets;
- build models;
- build Lightning modules;
- run training;
- perform wandb sweeps.

## Wandb Scope

Setup may configure wandb mode and login for dashboard logging.

Wandb sweeps are out of scope.
