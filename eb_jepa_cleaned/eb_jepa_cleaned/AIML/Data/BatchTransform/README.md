# AIML / Data / BatchTransform

The shared batch-transform contract. A utility/contract folder, not a buildable
object family.

## Files

- `base.py` — `BaseBatchTransform` (abstract). Owns the `__call__` flow
  (validate input -> `transform` -> validate output) and shared tensor/dict
  checks.

## The two transform families

`DataAugmentation` and `DataAdaptation` are **both** batch transforms: their
role bases (`BaseAugmentation`, `BaseAdaptation`) subclass `BaseBatchTransform`,
so every augmentation and every adaptation is a `BaseBatchTransform` and obeys
this same `dict -> dict` contract.

They are kept as separate families only to express intent (perturbation vs
representation adaptation) and to let configs select from the right registry.
**How, and in what order, augmentations and adaptations interact during
collation is entirely user-defined**: a concrete collator combines whichever
transforms it is given into its ordered `batch_transforms`, and the collator/
config decides the composition. The framework imposes no fixed ordering between
the two families.

## Why no registry / factory

`BaseBatchTransform` is an abstract contract, not a factory-buildable object.
The buildable transform families own the registries and factories; their
concrete objects subclass this base via their role bases. This folder therefore
has no `registry.py` / `factory.py`.

## Tests

`tests/unit/AIML/Data/BatchTransform/` — the call/validation flow and shared
checks.
