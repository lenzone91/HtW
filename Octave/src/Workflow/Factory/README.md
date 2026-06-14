# Workflow Factory README

This folder owns the shared registry-based factory foundation.

## File Roles

`registry.py`

- defines `Registry`, `RegistryEntry`, `SubBuildDeclaration`, and
  `FieldResolution`;
- stores declarative metadata about buildable objects;
- validates registrations early.

`builder.py`

- builds objects declared in a registry;
- supports single-object, named-object, and one-per-phase sub-builds;
- resolves declared fields before constructor invocation;
- copies configs before modifying them.

## Contract

Subsystem factories remain responsible for their own public wrappers. This
shared layer only provides the reusable mechanics for:

- registry lookup;
- type-field routing;
- config validation;
- field resolution;
- sub-object construction.

Objects built by this layer receive constructor-ready keyword arguments.
Runtime-dependent values must be made explicit through `runtime_context` and
declared `FieldResolution` objects.

## Migration Rule

Do not wire a subsystem to this layer until that subsystem has focused unit
tests covering:

- valid construction;
- unknown object names;
- unknown config keys;
- config non-mutation;
- any declared field resolution or sub-build behavior.
