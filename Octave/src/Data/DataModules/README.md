# DataModules README

This folder owns Lightning DataModule orchestration for AcVideoJepa.

DataModules assemble already-built phase objects:

```text
datasets + collators + dataloader_configs -> Lightning DataModule
```

They do not own raw sample adaptation or collation behavior.

## File Roles

```text
DataModules/
- base.py
- ac_video_jepa_datamodule.py
- registry.py
- factory.py
- configs.py
- README.md
```

`base.py` defines `BaseDataModule`.

It stores already-built datasets, already-built collators, and DataLoader configs by phase. It builds phase-specific PyTorch `DataLoader` objects through the Lightning dataloader hooks.

`ac_video_jepa_datamodule.py` defines `AcVideoJepaDataModule`.

It stays a thin AcVideoJepa-specific wrapper around `BaseDataModule` and registers itself with `DATAMODULE_REGISTRY` through the decorator-based factory API.

`registry.py` defines:

```text
DATAMODULE_REGISTRY
DATAMODULE_BUILDER
DATASETS_SUB_BUILD
COLLATORS_SUB_BUILD
```

It declares that DataModules depend on phase-keyed dataset and collator sub-builds.

`factory.py` exposes:

```text
build_ac_video_jepa_datamodule
build_datamodule_from_config
```

It imports DataModule implementation files so decorator registration is executed, validates DataModule runtime semantics, and delegates object construction to `src/Workflow/Factory`.

`configs.py` stores reusable plain DataModule config dictionaries.

## Factory Contract

The DataModule factory receives plain phase-keyed dictionaries:

```python
{
    "datasets": {
        "train": {
            "two_rooms": {
                "device": "cpu",
                "size": 2,
                "batch_size": 1,
            }
        },
        "val": None,
        "test": None,
    },
    "collators": {
        "train": {
            "ac_video_jepa": {
                "transforms": [],
            }
        },
        "val": None,
        "test": None,
    },
    "dataloader_configs": {
        "train": {
            "batch_size": 2,
            "shuffle": False,
            "num_workers": 0,
            "drop_last": False,
        },
        "val": None,
        "test": None,
    },
}
```

For each enabled phase, `datasets[phase]` must contain exactly one named dataset config.

For each enabled phase, `collators[phase]` must contain exactly one named collator config.

Disabled optional phases use `None`.

## Sub-build Contract

Datasets and collators are built through the owning child factories.

The DataModule registry declares:

```text
datasets  -> DATASET_BUILDER   with phase_single_named
collators -> COLLATOR_BUILDER  with phase_single_named
```

This means the shared factory layer builds one dataset and one collator per enabled phase before constructing the DataModule.

The DataModule factory must not instantiate dataset or collator classes directly.

## Phase Contract

Supported phases are:

```text
train
val
test
```

`train` should normally be enabled.

`val` and `test` may be disabled with `None`.

When an optional phase is disabled, the corresponding Lightning dataloader hook returns an empty list rather than `None`, because Lightning does not accept `None` from an implemented dataloader hook.

## Runtime Validation Contract

DataModules fail early for invalid runtime combinations:

* `persistent_workers=True` requires `num_workers > 0`;
* dataset `device: cuda` requires `torch.cuda.is_available()`;
* dataset `device: cuda` with `num_workers > 0` is rejected because EB-JEPA samples tensors directly on the requested device.

The CUDA worker restriction is a short-term project-level guard.

The long-term design is for datasets to emit CPU tensors and let Lightning move batches to the training device.

## Subsystem Contract

DataModules may:

* assemble already-built datasets and collators;
* own phase-specific DataLoader construction;
* validate DataLoader runtime semantics;
* validate phase consistency.

DataModules must not:

* adapt raw dataset samples;
* perform collation logic;
* resolve project paths;
* construct model objectives;
* know JEPA architecture details;
* run training.

## Registration Contract

DataModules are registered with the decorator-based registry API.

The expected pattern is:

1. define a thin DataModule class;
2. add its plain default config in `configs.py`;
3. register the class with `DATAMODULE_REGISTRY.register_class(...)`;
4. declare child dependencies through `SubBuildDeclaration`;
5. expose construction through `factory.py`.

Do not add manual DataModule builder registries in `factory.py`.

## Extension Steps

To add a DataModule:

1. define the DataModule class;
2. define its default plain config in `configs.py`;
3. register it with `DATAMODULE_REGISTRY`;
4. declare dataset/collator sub-builds if needed;
5. expose a public factory wrapper;
6. add focused tests for registration, phase consistency, sub-builds, and DataLoader smoke behavior;
7. update this README if the subsystem contract changes.

