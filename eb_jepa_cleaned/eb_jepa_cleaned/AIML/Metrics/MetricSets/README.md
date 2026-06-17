# AIML / Metrics / MetricSets

Generic metric-set machinery: collections of metrics with shared evaluation and
log-formatting.

## Files

- `metric_set.py` — `MetricSet` + `DEFAULT_METRIC_SET_CONFIG` + registration
  (name `"metric"`). Stores named metrics and evaluates each on its own inputs.
- `loggable_metric_set.py` — `LoggableMetricSet` + registration (name
  `"loggable"`). Extends `MetricSet` to flatten outputs into a logging-ready
  dict.
- `registry.py` — `METRIC_SET_REGISTRY` + `METRIC_SET_BUILDER` (routes by
  `set_type`, `check_default_keys=False`) and `METRICS_SUB_BUILD` (builds the
  named `metrics` via the metric builder).
- `factory.py` — thin `build_metric_set` (defaults `set_type` to `"loggable"`).

## Contract

A metric-set config carries `set_type` (routing) and a `metrics` mapping
(`{metric_name: metric_config}`); the sub-build turns `metrics` into built metric
objects before construction. Evaluating an unregistered metric raises (strict).
Non-scalar metric outputs require declared `known_output_names` on a subclass.

## Domain-agnostic

`MetricSet` and `LoggableMetricSet` are generic. Any metric set that wires a
specific experiment's fields automatically (e.g. context/target/prediction) is
experiment-specific and out of scope here; it belongs to the experiment pillar (AcVideoJEPA)
(Decision 29).

## Tests

`tests/unit/AIML/Metrics/MetricSets/` — evaluation dispatch, strict errors, and
log-dict flattening, with dummy generic metrics.
