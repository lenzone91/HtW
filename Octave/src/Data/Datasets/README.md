# Datasets README

This folder owns dataset-level data access for AcVideoJepa.

## File Roles

`two_rooms.py`

- wraps EB-JEPA `WallDataset`;
- adapts EB-JEPA `WallSample` objects into Octave semantic dictionaries;
- defines the AcVideoJepa sample key contract.

`configs.py`

- stores plain reusable dataset config dictionaries;
- exposes `DEFAULT_TWO_ROOMS_DATASET_CONFIG`;
- exposes `DEFAULT_DATASETS_CONFIG`.

`factory.py`

- builds Two Rooms datasets from plain dictionaries;
- converts plain config dictionaries into EB-JEPA `WallDatasetConfig`;
- dispatches named dataset configs.

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

- `states`: `[C, T, H, W]`;
- `actions`: `[2, T]`;
- `locations`: `[2, T]`;
- `wall_x`: scalar-like tensor;
- `door_y`: scalar-like tensor.

## Subsystem Contract

Datasets may adapt raw EB-JEPA samples into Octave dictionaries.

Dataset config adaptation belongs in `factory.py`, not in datasets.

Datasets must not:

- build DataLoaders;
- perform batch collation;
- resolve project paths;
- read run configs directly;
- construct Lightning objectives.

## Factory Contract

The dataset factory receives plain dictionaries:

```python
{
    "two_rooms": {
        "device": "cpu",
        "size": 2,
        ...
    }
}
```

The factory:

- copies user configs before modifying them;
- validates unknown keys early;
- merges user values with defaults;
- converts the merged dict into `WallDatasetConfig`;
- returns a `WallDatasetWrapper`.

`runtime_context` is accepted for interface compatibility but is not currently
needed by the Two Rooms dataset because no path resolution is required.

## Extension Steps

1. Add deterministic sample adaptation in the dataset wrapper.
2. Add plain defaults in `configs.py`.
3. Add construction logic in `factory.py`.
4. Keep the sample and factory contracts explicit in this README.
5. Add focused unit tests under `Octave/tests/unit`.
