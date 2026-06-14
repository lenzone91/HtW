# Datasets

This folder owns dataset-level data access for AcVideoJepa.

A dataset maps an integer sample index to a semantic sample dictionary consumed by collators and downstream training code.

```python
index -> sample_dict
```

## File Roles

`two_rooms.py`

* wraps EB-JEPA `WallDataset`;
* adapts EB-JEPA `WallSample` objects into Octave semantic dictionaries;
* defines the AcVideoJepa sample key contract;
* registers `WallDatasetWrapper` in `DATASET_REGISTRY` through the `src/Workflow/Factory` decorator API.

`configs.py`

* stores plain, serializable dataset config dictionaries;
* exposes `DEFAULT_TWO_ROOMS_DATASET_CONFIG`;
* exposes `DEFAULT_DATASETS_CONFIG`;
* does not instantiate dataset objects.

`registry.py`

* defines `DATASET_REGISTRY`;
* defines dataset-specific field resolutions;
* owns conversion from plain Octave config dictionaries to EB-JEPA `WallDatasetConfig`;
* exposes the shared registry builder used by this subsystem.

`factory.py`

* imports registered dataset implementations so decorator registration is executed;
* exposes public build functions;
* delegates object construction to the shared `src/Workflow/Factory` logic.

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

* `states`: `[C, T, H, W]`;
* `actions`: `[2, T]`;
* `locations`: `[2, T]`;
* `wall_x`: scalar-like tensor;
* `door_y`: scalar-like tensor.

`metadata` stores sample-level information and is intentionally kept as a dictionary at dataset level.

## Subsystem Contract

Datasets may:

* access individual samples;
* adapt external dataset objects into Octave semantic dictionaries;
* validate raw sample structure;
* preserve sample metadata.

Datasets must not:

* build DataLoaders;
* perform batch collation;
* resolve project paths;
* read run configs directly;
* construct models, losses, metrics, or Lightning modules;
* perform runtime orchestration.

## Factory Contract

Datasets are built from plain dictionary configs.

Single dataset config:

```python
{
    "dataset_type": "two_rooms",
    "device": "cpu",
    "size": 10000,
    ...
}
```

Named dataset configs:

```python
{
    "train": {
        "dataset_type": "two_rooms",
        "device": "cpu",
        "size": 10000,
        ...
    },
    "val": {
        "dataset_type": "two_rooms",
        "device": "cpu",
        "size": 1000,
        ...
    },
}
```

`dataset_type` selects the registered dataset implementation.

The dataset factory must not implement manual registry lookup or generic config merging. Those responsibilities belong to `src/Workflow/Factory`.

## EB-JEPA Config Adaptation

The Two Rooms dataset ultimately expects an EB-JEPA `WallDatasetConfig`.

Octave configs remain plain dictionaries. The conversion:

```python
dict -> WallDatasetConfig
```

belongs to `Data/Datasets/registry.py` as a dataset-specific field resolution.

This keeps:

* run configs serializable;
* dataset wrappers focused on sample access;
* generic factory logic centralized in `src/Workflow/Factory`.

## Registration Contract

New datasets should be registered with the decorator-based registry API.

The expected pattern is:

1. define the dataset wrapper;
2. add its plain default config in `configs.py`;
3. register the class with `DATASET_REGISTRY.register_class(...)`;
4. keep dataset-specific field adaptation in `registry.py`;
5. expose construction through `factory.py`.

Do not add manual `DATASET_BUILDERS_REGISTRY` dictionaries in `factory.py`.

## Validation Responsibility

Generic validation belongs to `src/Workflow/Factory`, including:

* selecting the object from its type field;
* checking config keys;
* merging defaults;
* applying field resolutions;
* instantiating the object.

Dataset-specific validation belongs here, including:

* converting plain configs into external dataset configs;
* checking external sample fields;
* enforcing the AcVideoJepa sample dictionary contract.

## Extension Steps

To add a new dataset:

1. create the dataset wrapper;
2. define its plain default config in `configs.py`;
3. register it with `DATASET_REGISTRY.register_class(...)`;
4. add dataset-specific field resolutions in `registry.py`;
5. import the dataset module in `factory.py` if needed to trigger registration;
6. add focused tests for registration, config validation, and sample contract;
7. update this README if the subsystem contract changes.

