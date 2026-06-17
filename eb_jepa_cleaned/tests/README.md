# tests

Test suite for the JEPA Framework. Pytest-style.

## Layout

    tests/
      unit/         one object, one local responsibility
      integration/  one subsystem flow
      smoke/        minimal end-to-end run

## Policy

- A migration step is not complete just because imports work. It is complete
  when the relevant tests and documentation are updated.
- Unit tests check local logic in isolation.
- Integration tests check interfaces between subsystems (the main source of
  bugs in this project).
- Smoke tests check minimal end-to-end execution.

Tests mirror the source pillar they cover (e.g. `unit/Workflow/...`).
