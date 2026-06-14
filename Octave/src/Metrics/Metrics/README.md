# Metrics README

This folder owns elementary metric modules for Octave.

Elementary metrics consume rollout outputs or rollout-derived tensors and return scalar metric values.

```text
metric_inputs -> scalar metric value
```

Metric composition belongs to `Metrics/MetricSets`.

Loss aggregation belongs to `Metrics/Loss`.

## File Roles

```text
Metrics/
- configs.py
- registry.py
- prediction_costs.py
- prediction_metrics.py
- regularizer_metrics.py
- factory.py
- README.md
```

`configs.py` stores plain metric and metric-helper configs.

`registry.py` defines:

```text
PREDICTION_COST_REGISTRY
PREDICTION_COST_BUILDER
METRIC_REGISTRY
METRIC_BUILDER
PREDICTION_COST_SUB_BUILD
```

`prediction_costs.py` defines registered prediction-cost modules.

`prediction_metrics.py` defines registered prediction-loss metric modules.

`regularizer_metrics.py` defines registered regularizer metric modules.

`factory.py` exposes:

```text
build_metric_from_config
build_metrics
```

It imports metric implementation files so decorator registration is executed, then delegates construction to `src/Workflow/Factory`.

## Factory Contract

Metrics are built from named metric configs.

```python
{
    "prediction_loss": {
        "metric_type": "autoregressive_prediction_loss",
        "prediction_cost": {
            "prediction_cost_type": "square_loss_seq",
            "use_projector": False,
            "proj": None,
        },
    },
    "std_loss": {
        "metric_type": "hinge_std",
        "projector": {
            "enabled": False,
            "mlp_spec": None,
            "hidden_multiplier": 4,
        },
        "std_margin": 1.0,
        "first_t_only": False,
        "spatial_as_samples": False,
    },
}
```

The outer key is the metric instance name.

The inner `metric_type` selects the registered metric class.

The metric factory must not implement manual metric registries or manual default merging. Generic registry lookup, config validation, default merging, field resolution, sub-build resolution, routing-field removal, and object construction belong to `src/Workflow/Factory`.

## Prediction Metrics

Prediction metrics compare encoded rollout targets with predicted latent states.

Registered prediction metrics:

```text
autoregressive_prediction_loss
parallel_prediction_loss
```

Prediction metrics receive a registered prediction cost through a sub-build.

Registered prediction costs:

```text
square_loss_seq
```

## Regularizer Metrics

Regularizer metrics wrap EB-JEPA regularization modules.

Registered regularizer metrics:

```text
hinge_std
covariance
temporal_similarity
inverse_dynamics
```

Regularizer metrics may require `encoder_shape` when building projection or inverse-dynamics helper modules.

`encoder_shape` is passed by the caller at factory time.

## Encoder Shape Contract

Some metric helpers depend on encoder output shape.

Expected fields:

```python
{
    "feature_dim": int,
    "height": int,
    "width": int,
}
```

This is required when a metric config enables:

```text
projector.enabled = True
inverse_dynamics_model.enabled = True
```

The metric subsystem may consume `encoder_shape`, but it must not compute it. Architecture/model factories own encoder probing.

## Subsystem Contract

Elementary metrics may:

* consume rollout outputs;
* compute scalar metric values;
* wrap EB-JEPA prediction costs and regularizers;
* build metric-local helper modules such as projectors or inverse dynamics models.

Elementary metrics must not:

* aggregate metrics into a log dictionary;
* compute the final training loss;
* build rollouts;
* build encoders, action encoders, or predictors;
* know Lightning step semantics;
* resolve paths.

## Registration Contract

Metrics are registered with the decorator-based registry API.

The expected pattern is:

1. define the metric or helper module;
2. add its plain default config in `configs.py`;
3. register it with the relevant registry;
4. declare sub-builds or field resolutions if constructor adaptation is required;
5. expose construction through `factory.py`.

Do not add manual metric dispatch dictionaries in `factory.py`.

## Extension Steps

To add a metric:

1. add a plain default config in `configs.py`;
2. create the metric module in `prediction_metrics.py`, `regularizer_metrics.py`, or a dedicated file;
3. register it with `METRIC_REGISTRY`;
4. add helper sub-builds or field resolutions if needed;
5. import the implementation module in `factory.py` so registration is executed;
6. add focused tests for registration, config validation, helper construction, and metric computation;
7. update this README if the subsystem contract changes.
