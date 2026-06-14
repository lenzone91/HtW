# Datasets README

This folder owns dataset-level sample access for AcVideoJepa.

Datasets operate on integer indices:

```text
index -> sample: dict
```

They expose semantic sample dictionaries consumed by collators and downstream training code.

## File Roles

```text
Datasets/
- two_rooms.py
- registry.py
- factory.py
- configs.py
- README.md
```

`two_rooms.py` defines `WallDatasetWrapper`.

It wraps EB-JEPA `WallDataset`, adapts EB-JEPA samples into Octave sample dictionaries, and registers the dataset with `DATASET_REGISTRY` through the decorator-based factory API.

`registry.py` defines:

```text
DATASET_REGISTRY
DATASET_BUILDER
```

It should stay declarative and should not contain dataset-specific conversion logic.

`factory.py` exposes public construction helpers:

```text
build_dataset_from_config
build_datasets
build_dataset
build_two_rooms_dataset
```

It imports dataset implementation files so decorator registration is executed, then delegates construction to the shared `src/Workflow/Factory` logic.

`configs.py` stores reusable plain dataset configs.

## Sample Contract

AcVideoJepa datasets return:

```python
{
    "states": tensor,
    "actions": tensor,
    "locations": tensor,
    "wall_x": tensor,
    "door_y": tensor,
    "metadata": {...},
}
```

The tensor shapes are inherited from EB-JEPA after its per-sample squeeze:

```text
states:    [C, T, H, W]
actions:   [2, T]
locations: [2, T]
wall_x:    scalar-like tensor
door_y:    scalar-like tensor
```

`metadata` stores sample-level information.

## Factory Contract

Datasets are built from named configs.

```python
{
    "two_rooms": {
        "device": "cpu",
        "size": 10000,
        ...
    }
}
```

The outer config key selects the registered dataset name.

There is no `dataset_type` field in the default named-config path.

The dataset factory must not implement manual dataset builder registries. Generic object construction belongs to `src/Workflow/Factory`.

## EB-JEPA Config Adaptation

Octave configs remain plain serializable dictionaries.

The Two Rooms dataset constructor expects an EB-JEPA `WallDatasetConfig`.

Therefore the conversion:

```text
dict -> WallDatasetConfig
```

is declared as a factory field resolution in the dataset registration decorator, next to `WallDatasetWrapper`.

The wrapper receives an already-prepared `WallDatasetConfig`.

## Registration Contract

Datasets are registered with the decorator-based registry API.

The expected pattern is:

1. define the dataset wrapper;
2. add its plain default config in `configs.py`;
3. register the class with `DATASET_REGISTRY.register_class(...)`;
4. declare constructor adaptations through field resolutions if needed;
5. expose construction through `factory.py`.

Do not add manual `DATASET_BUILDERS_REGISTRY` dictionaries.

Do not register classes manually inside `factory.py` with:

```python
DATASET_REGISTRY.register_class(...)(DatasetClass)
```

Registration should be attached to the class definition with a decorator.

## Subsystem Contract

Datasets may:

* access individual samples;
* adapt external dataset samples into Octave dictionaries;
* preserve sample metadata;
* validate raw sample structure;
* declare constructor-level config adaptation.

Datasets must not:

* build DataLoaders;
* perform batch collation;
* resolve project paths;
* construct models, losses, metrics, or Lightning modules;
* perform runtime orchestration.

## Validation Responsibility

Generic validation belongs to `src/Workflow/Factory`, including:

* selecting objects from registered names;
* checking config keys against default configs;
* merging defaults;
* applying field resolutions;
* instantiating objects.

Dataset-specific validation belongs here, including:

* converting plain configs into external dataset configs;
* checking external sample fields;
* enforcing the AcVideoJepa sample dictionary contract.

## Extension Steps

To add a dataset:

1. create the dataset wrapper;
2. define its plain default config in `configs.py`;
3. register it with `DATASET_REGISTRY.register_class(...)`;
4. add field resolutions if constructor adaptation is required;
5. import the implementation module in `factory.py` so registration is executed;
6. add focused tests for registration, config validation, and sample contract;
7. update this README if the subsystem contract changes.
