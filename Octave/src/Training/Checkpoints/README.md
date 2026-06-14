# Checkpoints README

This folder owns Lightning checkpoint callback construction.

Checkpoint configs are named callback configs:

```text
checkpoint_configs -> list[NamedModelCheckpoint]
```

The outer config key is the callback instance name. The inner `checkpoint_type` selects the registered checkpoint callback class.

## File Roles

```text
Checkpoints/
- base.py
- checkpoints.py
- registry.py
- factory.py
- configs.py
- README.md
```

`base.py` defines `NamedModelCheckpoint`.

`NamedModelCheckpoint` extends Lightning `ModelCheckpoint` by adding `checkpoint_name` to the callback state key. This prevents multiple checkpoint callbacks with similar Lightning state keys from colliding.

`checkpoints.py` defines registered checkpoint callback classes.

It registers checkpoint implementations through the decorator-based factory API and declares checkpoint-specific field resolutions, including runtime-aware `dirpath` resolution.

`registry.py` defines:

```text
CHECKPOINT_REGISTRY
CHECKPOINT_BUILDER
```

`factory.py` exposes:

```text
build_checkpoint_callbacks
build_checkpoint_callback
```

It imports checkpoint implementation files so decorator registration is executed, then delegates enabled callback construction to `src/Workflow/Factory`.

`configs.py` stores reusable plain checkpoint config dictionaries.

## Factory Contract

Checkpoint configs are plain dictionaries.

```python
{
    "last": {
        "checkpoint_type": "last",
        "dirpath": None,
        "filename": "last",
        "every_n_epochs": 1,
    }
}
```

The outer key, here `"last"`, is the callback instance name.

The inner `checkpoint_type` selects the registered checkpoint implementation.

The checkpoint factory must not implement manual checkpoint-type registries or manual default merging. Generic registry lookup, config validation, default merging, field resolution, and object construction belong to `src/Workflow/Factory`.

## Runtime Context Contract

Checkpoint directory resolution uses `runtime_context`.

If `dirpath` is `None`, the checkpoint callback uses:

```text
runtime_context["paths"]["checkpoints_dir"]
```

If `dirpath` is an absolute path, it is resolved directly.

If `dirpath` is relative, it is resolved from:

```text
runtime_context["paths"]["run_dir"]
```

This resolution belongs to the checkpoint subsystem because checkpoint callbacks own where their files are written.

## Registered Checkpoint Types

`last`

* saves the latest checkpoint;
* uses `save_last=True`;
* uses `save_top_k=0`.

`periodic`

* saves periodic checkpoints;
* uses `save_last=False`;
* uses `save_top_k=-1`.

`best_value`

* saves monitored best-value checkpoints;
* uses `save_last=False`;
* uses the configured `monitor`, `mode`, and `save_top_k`.

## Subsystem Contract

Checkpoints may:

* build Lightning checkpoint callbacks from plain configs;
* resolve callback `dirpath` through `runtime_context`;
* support last, periodic, and monitored best-value checkpoints;
* preserve unique callback state keys through `checkpoint_name`.

Checkpoints must not:

* run training;
* resume trainer state;
* load model weights;
* decide whether training should restart or continue;
* resolve paths outside checkpoint callback ownership.

## Resume Contract

Checkpoint callbacks save state.

Training resume belongs to `Execution`, usually through:

```python
trainer.fit(..., ckpt_path=...)
```

This folder does not decide resume policy.

## Registration Contract

Checkpoint callbacks are registered with the decorator-based registry API.

The expected pattern is:

1. define a checkpoint callback class;
2. add its plain default config in `configs.py`;
3. register the class with `CHECKPOINT_REGISTRY.register_class(...)`;
4. declare runtime-aware field resolutions if needed;
5. expose construction through `factory.py`.

Do not add manual `CHECKPOINT_DEFAULTS_BY_TYPE` or `CHECKPOINT_BUILDERS_REGISTRY` dictionaries in `factory.py`.

## Extension Steps

To add a checkpoint type:

1. add a plain default config in `configs.py`;
2. create a checkpoint callback class in `checkpoints.py`;
3. register it with `CHECKPOINT_REGISTRY.register_class(...)`;
4. add field resolutions if constructor adaptation is required;
5. import the implementation module in `factory.py` so registration is executed;
6. add focused tests for registration, config validation, path resolution, and callback construction;
7. update this README if the subsystem contract changes.

