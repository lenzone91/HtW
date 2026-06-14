# Metrics README

This folder owns metric modules, metric-set composition, and scalar loss aggregation for Octave.

The root `Metrics/` package is now a facade over three migrated subpackages:

```text
Metrics/
- Metrics/
- MetricSets/
- Loss/
```

The obsolete `ac_video_jepa_objective.py` path is not part of the migrated JEPA split logic.

## Folder Roles

`Metrics/Metrics/`

* owns elementary metric modules;
* owns prediction-cost helper modules;
* builds registered metric objects from plain configs.

`Metrics/MetricSets/`

* owns metric composition;
* routes structured runtime inputs to elementary metrics;
* returns flat metric dictionaries.

`Metrics/Loss/`

* owns scalar loss aggregation;
* consumes flat metric dictionaries;
* returns the final scalar loss and optional loss logs.

Root `Metrics/factory.py`

* exposes convenience wrappers around the migrated subfactories;
* builds the AcVideoJepa metric stack as `{metric_set, loss}`;
* does not build an `AcVideoJepaObjective`.

## Factory Contract

The root factory exposes:

```text
build_metrics
build_metric_from_config
build_metric_set
build_ac_video_jepa_metric_set
build_loss
build_ac_video_jepa_metric_stack
```

`build_ac_video_jepa_metric_stack(...)` returns:

```python
{
    "metric_set": metric_set,
    "loss": loss,
}
```

This is the object pair that the future `AcVideoJepaModule` refactor should receive directly.

## JEPA Split Contract

The migrated JEPA split separates responsibilities:

```text
rollout -> metric_set -> flat metric values -> loss
```

Therefore:

* rollout logic belongs to `Rollouts/`;
* elementary metric logic belongs to `Metrics/Metrics/`;
* metric routing belongs to `Metrics/MetricSets/`;
* scalar loss aggregation belongs to `Metrics/Loss/`;
* Lightning step orchestration belongs to `Models/Modules/`.

There should be no monolithic objective wrapper in the migrated path.

## Obsolete Objective Path

`ac_video_jepa_objective.py` is obsolete with respect to the JEPA split logic.

It should not be used for new code.

The root factory should not expose:

```text
build_ac_video_jepa_objective
```

The later `Models/Modules` migration should replace objective usage with already-built `metric_set` and `loss` objects.

## Subsystem Contract

Metrics code may:

* build elementary metrics;
* build metric sets;
* build scalar loss aggregators;
* wrap reusable EB-JEPA metric and regularizer primitives;
* produce flat metric dictionaries.

Metrics code must not:

* build encoders, action encoders, or predictors;
* implement rollout algorithms;
* own Lightning training steps;
* configure optimizers or schedulers;
* resolve paths or read run YAML files.

## Extension Steps

To add metric functionality:

1. add elementary metrics under `Metrics/Metrics/`;
2. add metric-set composition under `Metrics/MetricSets/`;
3. add scalar aggregation logic under `Metrics/Loss/`;
4. expose only stable convenience wrappers from root `Metrics/factory.py`;
5. update the relevant local README and this facade README if the public contract changes.

