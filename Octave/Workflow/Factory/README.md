# Workflow Factory README

This folder owns reusable factory abstractions for the Octave migration.

Layout note: new migration source code should be added under
`Octave/src/Workflow/Factory` once that folder is introduced. This folder
contains the initial migration draft and will be migrated incrementally.

## File Roles

`base.py`

- defines `BaseBuilder`;
- defines `BaseBuilderDispatcher`;
- centralizes common config validation;
- centralizes strict and non-strict error handling;
- propagates `runtime_context` through builders.

`__init__.py`

- exposes factory utilities when needed.

## Subsystem Contract

Factories build runtime objects from plain serializable dictionaries.

Allowed responsibilities:

- validate config structure;
- reject unknown keys when configured to do so;
- copy configs before mutation;
- pass `runtime_context` to child builders;
- dispatch named configs to registered builders.

Forbidden responsibilities:

- train or evaluate models;
- hide global state;
- resolve object-specific behavior;
- store experiment state.

## Runtime Context

Runtime-dependent facts belong in `runtime_context`.

Examples:

- resolved paths;
- run directories;
- device and hardware information;
- logger runtime state.

Builders may ignore `runtime_context` when they do not need it.

## Registry And Factory Details

`BaseBuilderDispatcher` expects:

```python
builder_registry[object_name] = builder
configs[object_name] = object_config
```

Concrete dispatchers may override this when the registry key is stored inside
the object config, but the override must remain construction-focused.

## Extension Steps

1. Define a default config dictionary.
2. Add a `BaseBuilder` subclass.
3. Implement `build_from_config`.
4. Register the builder in the owning subsystem.
5. Add or update focused tests.

## Ownership Rules

Reusable factory behavior belongs here.

AcVideoJepa-specific construction belongs in the owning subsystem:

- datasets in `Octave/Data/Datasets`;
- collators in `Octave/Data/Collators`;
- model components in `Octave/Model/ac_video_jepa`;
- Lightning orchestration in `Octave/Modules`.
