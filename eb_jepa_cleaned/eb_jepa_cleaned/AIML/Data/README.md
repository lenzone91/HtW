# AIML / Data

Generic, domain-agnostic data pipeline machinery.

## Subfolders

- `Datasets/` — the `{input, target, metadata}` sample convention and dataset
  build entrypoints.
- `BatchTransform/` — the shared `BaseBatchTransform` contract (util; no
  registry/factory).
- `DataAugmentation/` — stochastic data perturbation (e.g. random masking).
- `DataAdaptation/` — dataset-representation -> model-input adaptation
  (e.g. raw frames -> encoder input tensor); the dataset->model "interface".
- `Collators/` — list-of-samples -> batch, applying ordered batch transforms.
- `DataModules/` — Lightning DataModules composing datasets + collators per
  phase.

## Flow

    dataset sample {input, target, metadata}
      -> collator: collate list[sample] -> batch
      -> collator applies batch transforms (augmentations / adaptations)
      -> DataModule serves phase DataLoaders

## Transforms

`DataAugmentation` and `DataAdaptation` are **both** `BatchTransform`s (their
role bases subclass `BaseBatchTransform`). They are separate families only to
express intent and to drive config selection. **Their interaction and ordering
during collation is user-defined**: the collator/config composes them into the
collator's ordered `batch_transforms`; the framework imposes no fixed ordering.

## Dependency note

`DataAugmentation` and `DataAdaptation` share `BatchTransform` but not each
other. `DataModules` builds `Datasets` and `Collators` per phase via factory
sub-builds (wiring declared in `DataModules/registry.py`). All concrete
datasets/collators/augmentations/adaptations are experiment-specific
(Phase 4); this pillar holds only the machinery and the generic
`DefaultDataModule`.
