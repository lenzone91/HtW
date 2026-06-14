# AcVideoJepa Model README

This folder owns AcVideoJepa architecture component construction.

It builds the neural architecture objects used later by rollout, metrics, and Lightning modules.

It does not define JEPA training logic.

## File Roles

```text
ac_video_jepa/
- configs.py
- registry.py
- components.py
- factory.py
- README.md
```

`configs.py` stores plain reusable architecture config dictionaries.

`registry.py` defines:

```text
AC_VIDEO_JEPA_COMPONENTS_REGISTRY
AC_VIDEO_JEPA_COMPONENTS_BUILDER
```

`components.py` defines `AcVideoJepaComponents`.

It registers the architecture component bundle through the decorator-based factory API and owns construction of:

```text
encoder
action_encoder
predictor
encoder_shape
```

`factory.py` exposes:

```text
build_ac_video_jepa_components
```

It imports implementation files so decorator registration is executed, delegates construction to `src/Workflow/Factory`, and returns the historical dictionary API expected by downstream code.

## Factory Contract

The factory receives a plain serializable dictionary.

```python
{
    "model_type": "ac_video_jepa",
    "encoder": {
        "encoder_type": "impala",
        "width": 1,
        "stack_sizes": [16, 32, 32],
        "num_blocks": 2,
        "dropout_rate": None,
        "layer_norm": False,
        "input_channels": 2,
        "final_ln": True,
        "mlp_output_dim": 512,
        "input_shape": [2, 64, 64],
    },
    "action_encoder": {
        "action_encoder_type": "identity",
    },
    "predictor": {
        "predictor_type": "rnn",
        "hidden_size": None,
        "action_dim": 2,
        "num_layers": 1,
        "use_encoder_final_ln": True,
    },
}
```

`model_type` selects the registered architecture component bundle.

The generic registry lookup, config-key validation, routing-field removal, and object construction belong to `src/Workflow/Factory`.

## Returned Object Contract

The public factory returns a dictionary for compatibility with existing callers:

```python
{
    "encoder": encoder,
    "action_encoder": action_encoder,
    "predictor": predictor,
    "encoder_shape": encoder_shape,
}
```

Internally, the registry builds an `AcVideoJepaComponents` object, and `factory.py` converts it back to this dictionary through `as_dict()`.

## Encoder Shape Contract

`encoder_shape` is probed at architecture construction time.

Expected fields:

```python
{
    "feature_dim": int,
    "height": int,
    "width": int,
}
```

Downstream factories may consume this metadata, but they should not recompute it.

## Subsystem Contract

This folder may:

* instantiate EB-JEPA architecture classes;
* adapt YAML-friendly config values such as lists into tuple arguments;
* probe encoder output shape;
* return already-built architecture components.

This folder must not:

* expose a JEPA runtime object;
* parse batches;
* compute rollout trajectories;
* build prediction costs or regularizers;
* aggregate losses;
* construct optimizers or schedulers;
* log metrics;
* read run YAML directly;
* depend on global config state.

## Registration Contract

Architecture component bundles are registered with the decorator-based registry API.

The expected pattern is:

1. define the component-bundle class;
2. add its plain default config in `configs.py`;
3. register it with `AC_VIDEO_JEPA_COMPONENTS_REGISTRY.register_class(...)`;
4. expose construction through `factory.py`.

Do not add manual component-builder registries in `factory.py`.

## Extension Steps

To add a model architecture:

1. add a plain default config in `configs.py`;
2. create a component-bundle class;
3. register it with the architecture registry;
4. keep architecture-specific constructor adaptation inside the component-bundle class;
5. import the implementation module in `factory.py` so registration is executed;
6. add focused tests for registration, config validation, component construction, and encoder-shape probing;
7. update this README if the subsystem contract changes.

