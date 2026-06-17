# AIML / Data / DataAugmentation

Generic data-augmentation machinery. An augmentation stochastically perturbs a
batch (e.g. random masking or injecting noise). It is a batch transform applied
during collation.

## Files

- `base.py` — `BaseAugmentation(BaseBatchTransform)`, the augmentation family
  base (semantic marker; no config, no registration).
- `registry.py` — `AUGMENTATION_REGISTRY` + `AUGMENTATION_BUILDER` (anchor).
- `factory.py` — thin `build_augmentation` (one) / `build_augmentations`
  (ordered list).

## Contract

A concrete augmentation subclasses `BaseAugmentation` (which extends
`BaseBatchTransform` from `AIML/Data/BatchTransform/base.py`), implements
`transform(batch) -> batch`, and registers itself in its own object file.

`build_augmentations(configs, runtime_context=None)` returns an ordered list
(insertion order); an empty or absent config returns `[]`.

## Relationship to DataAdaptation

Augmentation and adaptation are both `dict -> dict` batch transforms sharing
`BaseBatchTransform`, but they are distinct families with separate registries:

- **DataAugmentation** — stochastic perturbation of data.
- **DataAdaptation** — dataset-representation -> model-input adaptation.

Both are `BaseBatchTransform`s. Their interaction and ordering during collation
is user-defined (the collator/config composes them); the framework imposes no
fixed ordering between the families.

## Domain-agnostic

This folder is generic. Concrete augmentations are experiment-specific and live
in the AcVideoJEPA pillar (Phase 4).

## Tests

`tests/unit/AIML/Data/DataAugmentation/` — the factory wrappers (exercised with
dummy generic augmentations).
