# DataModules README

This folder owns Lightning DataModule orchestration for AcVideoJepa.

## File Roles

`base.py`

- defines `BaseDataModule`;
- stores already-built datasets, collators, and DataLoader configs by phase;
- builds phase-specific PyTorch DataLoaders;
- returns empty lists for disabled optional validation/test phases.

`ac_video_jepa_datamodule.py`

- defines `AcVideoJepaDataModule`;
- stays a thin AcVideoJepa-specific wrapper around `BaseDataModule`.

`configs.py`

- stores plain default DataModule config dictionaries.

`factory.py`

- builds phase datasets through `Data/Datasets/factory.py`;
- builds phase collators through `Data/Collators/factory.py`;
- assembles `AcVideoJepaDataModule` from already-built objects.

## Subsystem Contract

DataModules own orchestration:

```text
datasets + collators + dataloader configs -> DataLoaders
```

DataModules must not:

- adapt raw dataset samples;
- perform collation logic;
- construct model objectives;
- know JEPA architecture details.

Optional `val` and `test` phases may be disabled with `None` datasets. The
Lightning hooks return empty lists for those disabled optional phases because
Lightning does not accept `None` from an implemented dataloader hook.

## Factory Contract

The factory receives plain phase-keyed dictionaries:

```python
{
    "datasets": {
        "train": {"dataset_type": "two_rooms", ...},
    },
    "collators": {
        "train": {"collator_type": "ac_video_jepa", ...},
    },
    "dataloader_configs": {
        "train": {"batch_size": 2, "shuffle": False},
    },
}
```

`dataset_type` defaults to `two_rooms` when omitted because AcVideoJepa is the
only supported migration target.

## Extension Steps

1. Keep DataModule classes thin.
2. Add child-object construction to `factory.py`.
3. Validate phase consistency early.
4. Add a DataLoader smoke test.
5. Update this README when orchestration changes.
