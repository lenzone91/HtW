# Workflow README

This folder contains legacy reusable workflow helpers.

## Folder Roles

`Factory/`

- provides reusable builder and dispatcher abstractions;
- documents factory validation and registry contracts.

## Subsystem Contract

Workflow helpers may:

- provide reusable construction utilities;
- validate plain config dictionaries;
- propagate `runtime_context` through builders.

Workflow helpers must not:

- own AcVideoJepa-specific construction;
- run training or validation;
- resolve experiment paths directly.

## Extension Steps

1. Prefer adding new migration code under `Octave/src`.
2. Update `Factory/` only for reusable factory behavior.
3. Add focused tests when behavior changes.
4. Update this README and the local subfolder README.

## Ownership Rules

Reusable framework helpers belong here only while this legacy folder remains.

AcVideoJepa-specific code belongs under `Octave/src`.
