# Configs README

This folder will hold future AcVideoJepa run configs.

Configs should remain plain and serializable.

Runtime-resolved facts, such as absolute run directories or machine-specific
paths, belong in `runtime_context`, not in model, dataset, or module objects.

## Folder Roles

`runs/`

- stores runnable YAML configs;
- starts with a tiny AcVideoJepa toy-run config for smoke testing.
