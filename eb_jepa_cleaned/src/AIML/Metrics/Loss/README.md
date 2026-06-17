# AIML / Metrics / Loss

Generic loss machinery: a training loss assembled from metric values.

## Files

- `loss.py` — `WeightedMetricLoss` + `DEFAULT_WEIGHTED_METRIC_LOSS_CONFIG` +
  registration (name `"weighted_metric"`, routed by `loss_type`).
- `registry.py` — `LOSS_REGISTRY` + `LOSS_BUILDER` (anchor).
- `factory.py` — thin `build_loss` (imports `loss` for registration).

## WeightedMetricLoss

Takes a flat `metric_weights` dict and returns a weighted sum (to minimize):
positive weight -> minimize the metric, negative -> maximize, 0/None -> inactive
(dropped at construction). It is generic: it operates on a dict of scalar metric
values and knows nothing about the domain. A required metric missing at forward
time raises (strict).

## Tests

`tests/unit/AIML/Metrics/Loss/` — weighting, filtering, strict errors.
