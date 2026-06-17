# AIML / Data / DataAdaptation

Generic data-adaptation machinery. An adaptation maps a dataset's output
representation to a model's input representation (e.g. raw frames -> the tensor
an encoder expects). These are the dataset->model "interface" transforms.

## Files

- `base.py` — `BaseAdaptation(BaseBatchTransform)`, the adaptation family base
  (semantic marker; no config, no registration). The representation-contract
  helpers will grow here in a later phase.
- `registry.py` — `ADAPTATION_REGISTRY` + `ADAPTATION_BUILDER` (anchor).
- `factory.py` — thin `build_adaptation` (one) / `build_adaptations`
  (ordered list).

## Contract

A concrete adaptation subclasses `BaseAdaptation` (which extends
`BaseBatchTransform` from `AIML/Data/BatchTransform/base.py`), implements
`transform(batch) -> batch`, and registers itself in its own object file.

`build_adaptations(configs, runtime_context=None)` returns an ordered list
(insertion order); an empty or absent config returns `[]`.

## Relationship to DataAugmentation

Adaptation and augmentation are both `dict -> dict` batch transforms sharing
`BaseBatchTransform`, but they are distinct families with separate registries:

- **DataAdaptation** — representation adaptation (dataset output -> model input).
- **DataAugmentation** — stochastic perturbation of data.

Both are `BaseBatchTransform`s. Their interaction and ordering during collation
is user-defined (the collator/config composes them); the framework imposes no
fixed ordering between the families.

The representation-adaptation contract (declaring source/target representations
and propagating metadata) will grow on `BaseAdaptation` when the concrete
experiment adaptations land (Phase 4). The exact pipeline stage at which
adaptation is applied (collator vs datamodule) is decided when the
collator/datamodule are assembled (A2).

## Domain-agnostic

This folder is generic. Concrete adaptations (e.g. raw frames -> encoder input)
are experiment-specific and live in the AcVideoJEPA pillar (Phase 4).

## Tests

`tests/unit/AIML/Data/DataAdaptation/` — the factory wrappers (exercised with
dummy generic adaptations).
