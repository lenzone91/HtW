# AIML / Metrics

Generic, domain-agnostic metric machinery.

## Subfolders

- `Metrics/` — individual metrics: `BaseMetric` + registry/factory. Concrete
  metrics are experiment-specific (Phase 4).
- `Loss/` — `WeightedMetricLoss`: a training loss assembled from metric values.
- `MetricSets/` — `MetricSet` / `LoggableMetricSet`: collections of metrics with
  shared evaluation and log formatting.

## Composition

    metric configs --build_metric(s)--> metric objects
    metric-set config (metrics: {...}) --build_metric_set--> MetricSet
        (metrics sub-built via the metric builder)
    metric_weights --build_loss--> WeightedMetricLoss

`factory.py` re-exports `build_metric`, `build_metrics`, `build_loss`,
`build_metric_set`.

## Experiment concretes

The JEPA objective metrics (latent prediction loss + anti-collapse regularizers)
and any other experiment-specific metrics register from the AcVideoJEPA pillar
(Phase 4). AIML owns only the generic `MetricSet` / `LoggableMetricSet`
(Decisions 26/33).

## Tests

`tests/unit/AIML/Metrics/...` for each family; metric-set evaluation and the
metrics sub-build are exercised with dummy generic metrics.
