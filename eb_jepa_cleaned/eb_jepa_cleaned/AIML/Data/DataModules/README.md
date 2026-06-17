# AIML / Data / DataModules

Generic Lightning DataModule machinery and the default concrete DataModule.

## Files

- `base.py` — `BaseDataModule` (abstract). Stores one dataset, one collator, and
  one DataLoader config per phase (train/val/test) and builds phase DataLoaders.
  Receives already-built datasets and collators; it never builds them.
- `registry.py` — `DATAMODULE_REGISTRY` + `DATAMODULE_BUILDER` and the
  cross-subsystem `DATAMODULE_SUB_BUILDS` (datasets and collators are built
  per phase via `phase_single_named`).
- `default.py` — `DefaultDataModule` + `DEFAULT_DATAMODULE_CONFIG` +
  registration (name `"default"`). The one generic concrete DataModule.
- `factory.py` — thin `build_datamodule` (exactly one) / `build_datamodules`.

## Config shape

    {
      "default": {
        "datasets":           {"train": {<name>: {...}}, "val": ..., "test": ...},
        "collators":          {"train": {<name>: {...}}, "val": ..., "test": ...},
        "dataloader_configs": {"train": {...}, "val": {...}, "test": {...}},
      }
    }

Each phase under `datasets` / `collators` is a single named entry (built via
`phase_single_named`); a phase set to `None` yields a `None` object for that
phase (e.g. no test set). `DEFAULT_DATAMODULE_CONFIG` is only the top-level key
allow-list; per-phase contents come from the composed config.

## Domain-agnostic

`DefaultDataModule` is generic: it composes whichever registered datasets and
collators the config selects. The datasets and collators themselves are
experiment-specific (Phase 4).

## Tests

- `tests/unit/AIML/Data/DataModules/` — `BaseDataModule` phase logic and the
  factory building `DefaultDataModule` (with dummies).
- `tests/integration/AIML/Data/` — the full data flow: config dict -> factory ->
  DataModule -> DataLoader -> batch (dataset -> collator -> augmentation).
