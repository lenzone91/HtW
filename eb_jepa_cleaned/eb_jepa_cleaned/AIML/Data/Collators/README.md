# AIML / Data / Collators

Generic collator machinery. A collator maps a list of samples to a batch and
applies an ordered list of batch transforms.

## Files

- `base.py` — `BaseCollator` (abstract). Owns the flow (validate samples ->
  `collate_samples` -> apply `batch_transforms` -> validate batch). No default
  config, no registration.
- `registry.py` — `COLLATOR_REGISTRY` + `COLLATOR_BUILDER` (anchor).
- `factory.py` — thin `build_collator` / `build_collators`.

## Contract

A concrete collator subclasses `BaseCollator`, implements
`collate_samples(samples) -> batch`, and registers itself in its own object
file.

`batch_transforms` is family-agnostic: an ordered list of `BaseBatchTransform`.
Both `DataAugmentation` and `DataAdaptation` objects are `BaseBatchTransform`s,
so a collator can hold either or both.

**The interaction between augmentation and adaptation during collation is
user-defined.** A concrete collator declares its own batch-transform `sub_builds`
(to `AUGMENTATION_BUILDER` / `ADAPTATION_BUILDER`), and the collator/config
decides which transforms are included and in what order they apply. The
framework imposes no fixed ordering between the two families. This wiring lives
with the concrete experiment collator (Phase 4).

## Domain-agnostic

This folder is generic. Concrete collators are experiment-specific and live in
the AcVideoJEPA pillar (Phase 4).

## Tests

`tests/unit/AIML/Data/Collators/` — the collate/transform flow and the factory
(exercised with a dummy generic collator).
