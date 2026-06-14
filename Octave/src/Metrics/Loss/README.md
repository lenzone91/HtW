# Loss README

This folder owns scalar loss aggregation for Octave metrics.

Loss objects consume flat metric dictionaries:

```text
metric_values: dict[str, scalar tensor] -> loss, loss_logs
```

They do not compute metrics themselves.

## File Roles

```text
Loss/
- configs.py
- registry.py
- loss.py
- factory.py
- README.md
```

`configs.py` stores plain loss config dictionaries.

`registry.py` defines:

```text
LOSS_REGISTRY
LOSS_BUILDER
```

`loss.py` defines registered loss modules.

`factory.py` exposes:

```text
build_loss
```

It imports loss implementation files so decorator registration is executed, then delegates construction to `src/Workflow/Factory`.

## Factory Contract

Losses are built from plain configs:

```python
{
    "loss_type": "weighted_metric",
    "metric_weights": {
        "prediction_loss": 1.0,
        "std_loss": 16.0,
    },
    "strict": True,
    "name": "loss",
    "return_loss_terms": True,
}
```

`loss_type` selects the registered loss implementation.

The loss factory must not implement manual loss registries or manual default merging. Generic registry lookup, config validation, default merging, routing-field removal, and object construction belong to `src/Workflow/Factory`.

## Weighted Metric Loss

`WeightedMetricLoss` computes a weighted sum over scalar metric values.

Weights set to `None` or `0` are ignored.

When `return_loss_terms=True`, the returned logs include:

```text
loss
loss/<metric_name>
```

When `return_loss_terms=False`, only the total loss is returned.

## Subsystem Contract

Loss modules may:

* consume flat metric dictionaries;
* combine scalar metrics into one scalar optimization loss;
* return loss logs.

Loss modules must not:

* compute rollout metrics;
* know JEPA rollout structure;
* build metric sets;
* build models or modules;
* perform logging side effects.

## Registration Contract

Losses are registered with the decorator-based registry API.

The expected pattern is:

1. define the loss module;
2. add its plain default config in `configs.py`;
3. register it with `LOSS_REGISTRY.register_class(...)`;
4. expose construction through `factory.py`.

Do not add manual loss registry dictionaries in `factory.py`.

## Extension Steps

To add a loss:

1. add a plain default config in `configs.py`;
2. create the loss module in `loss.py` or a dedicated file;
3. register it with `LOSS_REGISTRY`;
4. import the implementation module in `factory.py` so registration is executed;
5. add focused tests for registration, config validation, and loss computation;
6. update this README if the subsystem contract changes.
