# AIML / Data / Datasets

Generic dataset machinery: the sample convention and the build entrypoints.

## Files

- `base.py` — `BaseDataset` (abstract). Owns the generic sample convention
  `{input, target, metadata}` and shared helpers (`build_sample`,
  `check_sample`, `to_tensor`). No default config, no registration.
- `registry.py` — `DATASET_REGISTRY` + `DATASET_BUILDER` (the import anchor).
- `factory.py` — thin `build_dataset` / `build_datasets`.

## Contract

A concrete dataset subclasses `BaseDataset`, yields samples following the
`{input, target, metadata}` convention, and registers itself in its own object
file (`@DATASET_REGISTRY.register_class(name=..., default_config=...)`).

`build_dataset(config, name, runtime_context=None)` builds one by name.
`build_datasets(configs, runtime_context=None)` builds a named dict.

## Domain-agnostic

This folder is generic. Concrete datasets (e.g. the two-rooms video dataset) are
experiment-specific and live in the AcVideoJEPA pillar (Phase 4); they register
onto `DATASET_REGISTRY` at import time.

## Deferred

Shared dataset field resolvers from the prior framework (path resolution via
`runtime_context['data']['dataset_roots']`, and serializable-dtype resolution)
are intentionally not ported yet:

- the path resolver depends on the Setup runtime-context contract, which is
  deferred (Decision 22);
- both are declared per concrete dataset entry, and no generic concrete dataset
  exists yet.

They will land in Phase 4 alongside the concrete datasets that use them.

## Tests

`tests/unit/AIML/Data/Datasets/` — `BaseDataset` convention/validation and the
factory wrappers (exercised with a dummy generic dataset).
