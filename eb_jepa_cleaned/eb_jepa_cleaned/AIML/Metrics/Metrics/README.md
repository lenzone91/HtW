# AIML / Metrics / Metrics

Generic metric machinery: the metric base and build entrypoints.

## Files

- `base.py` — `BaseMetric` (abstract). Owns reduction handling, the
  `check_inputs` hook, shared shape/autograd checks, and a NumPy helper. No
  default config, no registration.
- `registry.py` — `METRIC_REGISTRY` + `METRIC_BUILDER` (anchor).
- `factory.py` — thin `build_metric` / `build_metrics`.

## Contract

A concrete metric subclasses `BaseMetric`, implements its computation (and
optionally overrides `check_inputs`), owns its `*_DEFAULT_CONFIG`, and registers
itself in its own object file.

## Domain-agnostic

Concrete metrics are experiment-specific: the JEPA objective metrics (latent
prediction error and anti-collapse regularizers) and any evaluation metrics
register from the AcVideoJEPA pillar (Phase 4). `BaseMetric` here stays
modality-agnostic.

## Tests

`tests/unit/AIML/Metrics/Metrics/` — `BaseMetric` reduction/validation and the
factory (with a dummy generic metric).
