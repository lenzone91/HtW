# AcVideoJEPA / Data

The experiment's data: the two-rooms dataset and the collator that batches it.

## Layout

- `two_rooms/` — the **vendored** two-rooms dataset/environment, copied verbatim
  from EB-JEPA (Decision 30). Treat as third-party; wrap, don't edit. Deps
  (`acvideo` extra): scipy, gymnasium, pyyaml, numpy, torch.
- `Datasets/` — `TwoRoomsDataset`, a semantic-dictionary adapter over the
  vendored `WallDataset`, registered `two_rooms` on the AIML dataset registry.
- `Collators/` — `AcVideoJepaCollator`, registered `ac_video_jepa` on the AIML
  collator registry.

## Sample / batch schema

JEPA is self-supervised, so samples do **not** use the generic
`{input, target}` convention. A sample is
`{states, actions, locations, wall_x, door_y, metadata}`; the collator stacks the
tensor fields into a batch of the same keys (metadata kept as a list). The module
reads `batch["states"]` and `batch["actions"]`.

## DataModule

No experiment-specific datamodule: AIML's generic `DefaultDataModule` composes
the dataset + collator + per-phase DataLoader configs from config.

## Build configs

The dataset build config is flat (the `WallDatasetConfig` fields); the adapter
assembles the `WallDatasetConfig`. The builder does not merge defaults — omitted
fields fall back to the dataclass defaults.

## Tests

- `tests/unit/AcVideoJEPA/Data/` — building the dataset/collator through the AIML
  factories and producing a batch (tiny config), plus the full
  dataset → collator → DataLoader flow.
