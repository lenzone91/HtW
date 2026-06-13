# Data README

This folder owns AcVideoJepa data pipeline code.

## Folder Roles

`Datasets/`

- wraps EB-JEPA datasets;
- adapts samples to Octave semantic dictionaries.

`Collators/`

- batches AcVideoJepa semantic samples;
- preserves metadata;
- exposes a config-driven collator factory.

`DataModules/`

- assembles already-built datasets and collators;
- owns phase-specific DataLoader construction;
- exposes a config-driven AcVideoJepa DataModule factory.

Future folders:

- `DataTransforms/`.

## Ownership Rules

Datasets return samples.

Collators create batches.

DataModules create DataLoaders from already-built datasets and collators.
