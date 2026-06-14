# Loading README

This folder owns weight loading into already-built runtime objects.

Loading restores weights into objects that have already been constructed elsewhere:

```text
already-built module + loading config -> module with restored weights
```

Loading does not build models and does not resume Trainer state.

## File Roles

```text
Loading/
- configs.py
- registry.py
- module_loading.py
- factory.py
- README.md
```

`configs.py` stores plain loading config dictionaries.

`registry.py` defines:

```text
LOADING_REGISTRY
LOADING_BUILDER
```

`module_loading.py` defines:

```text
load_module_from_lightning_checkpoint
LightningModuleLoader
```

`LightningModuleLoader` is registered through the decorator-based factory API. It is a callable loading policy that restores weights into an already-built LightningModule.

`factory.py` exposes:

```text
build_module_loader
build_default_module_loader
load_module_if_needed
is_loading_enabled
```

It imports loading implementation files so decorator registration is executed, handles disabled loading configs, and delegates enabled loading-policy construction to `src/Workflow/Factory`.

## Factory Contract

Disabled loading config:

```python
{
    "enabled": False,
    "type": "lightning_module",
    "checkpoint_path": None,
    "strict": True,
    "map_location": "cpu",
    "state_dict_key": "state_dict",
    "relative_to": "run_dir",
}
```

Enabled loading config:

```python
{
    "enabled": True,
    "type": "lightning_module",
    "checkpoint_path": "checkpoints/last.ckpt",
    "strict": True,
    "map_location": "cpu",
    "state_dict_key": "state_dict",
    "relative_to": "run_dir",
}
```

`type` selects the registered loading-policy class.

The loading factory must not implement manual loading-type dispatch or manual default merging. Generic registry lookup, config validation, default merging, field resolution, routing-field removal, and object construction belong to `src/Workflow/Factory`.

## Runtime Context Contract

Relative checkpoint paths are resolved through `runtime_context`.

If `relative_to` is `"run_dir"`, the checkpoint path is resolved from:

```text
runtime_context["paths"]["run_dir"]
```

Other path roots are allowed if they exist in:

```text
runtime_context["paths"]
```

Absolute checkpoint paths are resolved directly.

This path resolution belongs to Loading because this subsystem owns where weight files are read from.

## Delayed Loading Contract

`build_module_loader(...)` returns a callable object.

The returned object is later called with an already-built module:

```python
module_loader(module)
```

For enabled loading configs, this restores weights from the configured checkpoint.

For disabled loading configs, this returns the module unchanged.

## Loading Vs Resume

Loading restores weights into an already-built module.

Resume belongs to `Execution/train.py` and uses:

```python
trainer.fit(..., ckpt_path=...)
```

Use Loading for validation/evaluation weight restoration. Use resume to continue interrupted training.

## Subsystem Contract

Loading may:

* restore weights into already-built modules;
* resolve checkpoint paths through `runtime_context`;
* support validation/evaluation loading;
* validate checkpoint state-dict keys.

Loading must not:

* build models or modules;
* build optimizers or schedulers;
* restore optimizer or scheduler state;
* resume Trainer state;
* run training.

## Registration Contract

Loading policies are registered with the decorator-based registry API.

The expected pattern is:

1. define a callable loading-policy class;
2. add its plain default config in `configs.py`;
3. register it with `LOADING_REGISTRY.register_class(...)`;
4. declare checkpoint-path field resolution if needed;
5. expose construction through `factory.py`.

Do not add manual loading-type registries in `factory.py`.

## Extension Steps

To add a loading policy:

1. add a plain default config in `configs.py`;
2. create a callable loading-policy class;
3. register it with `LOADING_REGISTRY`;
4. add field resolutions for checkpoint-path adaptation if needed;
5. import the implementation module in `factory.py` so registration is executed;
6. add focused tests for registration, disabled behavior, path resolution, and weight restoration;
7. update this README if the subsystem contract changes.

