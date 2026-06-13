# Collators README

This folder owns batching for AcVideoJepa samples.

## File Roles

`base.py`

- defines `BaseBatchTransform`;
- defines `BaseCollator`;
- validates sample lists and batch dictionaries;
- applies transforms after collation.

`ac_video_jepa_collator.py`

- defines `AcVideoJepaCollator`;
- stacks AcVideoJepa sample tensors;
- preserves sample metadata as a list.

`configs.py`

- stores plain collator config dictionaries.

`factory.py`

- builds collators from plain configs;
- dispatches by `collator_type`.

## Batch Contract

The AcVideoJepa collator maps:

```python
list[sample_dict] -> batch_dict
```

Input samples must follow the dataset contract:

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

Output batches use the same semantic keys:

```python
{
    "states": tensor,
    "actions": tensor,
    "locations": tensor,
    "wall_x": tensor,
    "door_y": tensor,
    "metadata": list[dict],
}
```

## Subsystem Contract

Collators may:

- stack tensors along the batch dimension;
- preserve metadata;
- apply already-built transforms.

Collators must not:

- load data;
- resolve paths;
- construct model targets;
- depend on Lightning.

## Factory Contract

The collator factory receives plain dictionaries:

```python
{
    "collator_type": "ac_video_jepa",
    "transforms": [],
}
```

`transforms` currently accepts already-built transform objects. A transform
factory can be added later when transform configs are needed.

## Extension Steps

1. Add the collator class.
2. Add plain defaults in `configs.py`.
3. Register the builder in `factory.py`.
4. Add focused unit tests.
5. Update this README when the batch contract changes.
