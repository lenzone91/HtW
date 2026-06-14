# MetricSets README

This folder owns metric composition for Octave.

A metric set groups already-built elementary metric modules and routes structured runtime inputs to each metric.

```text
rollout outputs -> metric set -> flat metric dictionary
```

Metric sets do not compute the final scalar optimization loss. Loss aggregation belongs to `Metrics/Loss`.

## File Roles

```text
MetricSets/
- configs.py
- registry.py
- metric_set.py
- factory.py
- README.md
```

`configs.py` stores plain metric-set config dictionaries.

`registry.py` defines:

```text
METRIC_SET_REGISTRY
METRIC_SET_BUILDER
METRICS_SUB_BUILD
```

It declares that metric sets depend on named elementary metric sub-builds.

`metric_set.py` defines metric-set classes.

It registers `AcVideoJepaMetricSet` through the decorator-based factory API.

`factory.py` exposes:

```text
build_metric_set
build_ac_video_jepa_metric_set
```

It imports metric-set implementation files so decorator registration is executed, then delegates construction to `src/Workflow/Factory`.

## Factory Contract

Metric sets are built from plain configs:

```python
{
    "set_type": "ac_video_jepa",
    "strict": True,
    "metric_to_input_names": None,
    "metrics": {
        "prediction_loss": {
            "metric_type": "autoregressive_prediction_loss",
            "prediction_cost": {
                "prediction_cost_type": "square_loss_seq",
                "use_projector": False,
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
    },
}
```

`set_type` selects the registered metric-set class.

The `metrics` section is a named metric config dictionary. Each inner key is the metric instance name.

The metric-set factory must not manually instantiate elementary metrics. Metric construction belongs to `Metrics/Metrics`.

## Sub-build Contract

Metric sets declare:

```text
metrics -> METRIC_BUILDER with build_method="named"
```

The shared factory layer builds all elementary metrics before constructing the metric set.

The resulting constructor keyword is:

```python
metrics: dict[str, nn.Module]
```

Therefore `MetricSet` accepts both:

```python
metrics={...}
```

and direct keyword metrics:

```python
prediction_loss=...
std_loss=...
```

This keeps compatibility with manual construction while supporting registry sub-builds.

## AcVideoJepa Routing Contract

`AcVideoJepaMetricSet` routes available runtime inputs to metric-specific inputs.

Default routes:

```python
{
    "prediction_loss": ("rollout_output",),
    "std_loss": ("rollout_output",),
    "cov_loss": ("rollout_output",),
    "sim_loss_t": ("rollout_output",),
    "idm_loss": ("rollout_output", "actions"),
}
```

Valid input names are:

```text
rollout_output
actions
```

Metric outputs are converted into a flat log dictionary.

## Subsystem Contract

Metric sets may:

* store elementary metric modules;
* route structured inputs to metric modules;
* call metric modules;
* flatten metric outputs into log dictionaries;
* validate metric routes.

Metric sets must not:

* build elementary metrics manually;
* compute the final scalar training loss;
* build rollouts;
* build models or modules;
* resolve paths;
* perform logging side effects.

## Registration Contract

Metric sets are registered with the decorator-based registry API.

The expected pattern is:

1. define the metric-set class;
2. add its plain default config in `configs.py`;
3. register it with `METRIC_SET_REGISTRY.register_class(...)`;
4. declare metric sub-builds through `SubBuildDeclaration`;
5. expose construction through `factory.py`.

Do not add manual metric-set builder registries in `factory.py`.

## Extension Steps

To add a metric set:

1. add a plain default config in `configs.py`;
2. create the metric-set class in `metric_set.py` or a dedicated file;
3. register it with `METRIC_SET_REGISTRY`;
4. declare metric sub-builds if needed;
5. import the implementation module in `factory.py` so registration is executed;
6. add focused tests for registration, sub-builds, route validation, and flat log formatting;
7. update this README if the subsystem contract changes.
