# Collators

This folder owns batching for AcVideoJepa samples.

A collator maps a list of semantic sample dictionaries produced by a dataset into one batch dictionary consumed by the training pipeline.

```python
list[sample_dict] -> batch_dict
```

## File Roles

`base.py`

* defines `BaseBatchTransform`;
* defines `BaseCollator`;
* validates input sample lists;
* validates output batch dictionaries;
* applies already-built batch transforms after collation.

`ac_video_jepa_collator.py`

* defines `AcVideoJepaCollator`;
* stacks AcVideoJepa tensor fields along the batch dimension;
* preserves sample metadata as a list;
* registers the collator in `COLLATOR_REGISTRY` through the `src/Workflow/Factory` decorator API.

`configs.py`

* stores plain, serializable default collator configs;
* does not instantiate Python objects.

`registry.py`

* defines `COLLATOR_REGISTRY`;
* defines collator-specific field resolutions;
* owns collator-domain config adaptation and validation;
* exposes the shared registry builder used by this subsystem.

`factory.py`

* imports registered collator implementations so decorator registration is executed;
* exposes public build functions;
* delegates object construction to the shared `src/Workflow/Factory` logic.

## Batch Contract

Input samples must follow the AcVideoJepa dataset sample contract:

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

The collated batch keeps the same semantic fields:

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

Tensor fields are stacked with a new batch dimension. Metadata is not stacked because it may contain non-tensor sample-level information.

## Subsystem Contract

Collators may:

* stack sample tensors into batch tensors;
* preserve sample metadata;
* apply already-built batch transforms;
* validate collator-specific batch semantics.

Collators must not:

* load data;
* resolve file paths;
* build datasets;
* build model targets;
* depend on Lightning;
* perform runtime orchestration.

## Factory Contract

Collators are built from plain dictionary configs.

```python
{
    "collator_type": "ac_video_jepa",
    "transforms": [],
}
```

`collator_type` selects the registered collator implementation.

`transforms` currently contains already-built callable batch transforms. A transform factory can be added later if transform configs become necessary.

The collator factory must not implement manual registry lookup or generic config merging. Those responsibilities belong to `src/Workflow/Factory`.

## Registration Contract

New collators should be registered with the decorator-based registry API.

The expected pattern is:

1. define the collator class;
2. add its plain default config in `configs.py`;
3. register the class with `COLLATOR_REGISTRY.register_class(...)`;
4. keep collator-specific field validation in `registry.py`;
5. expose construction through `factory.py`.

Do not add manual `COLLATOR_BUILDERS_REGISTRY` dictionaries in `factory.py`.

## Validation Responsibility

Generic validation belongs to `src/Workflow/Factory`, including:

* selecting the object from its type field;
* checking config keys;
* merging defaults;
* applying field resolutions;
* instantiating the object.

Collator-specific validation belongs here, including:

* checking that `transforms` is a list;
* checking that every transform is callable;
* checking AcVideoJepa sample keys;
* checking tensor-valued sample fields.

## Extension Steps

To add a new collator:

1. create the collator class;
2. define its default plain config in `configs.py`;
3. register it with `COLLATOR_REGISTRY.register_class(...)`;
4. add any collator-specific field resolution in `registry.py`;
5. import the collator module in `factory.py` if needed to trigger registration;
6. add focused tests for registration, config validation, and batch contract;
7. update this README if the subsystem contract changes.

