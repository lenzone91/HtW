# Metrics README

This folder owns objective, loss, regularizer, and metric composition for
Octave runtime objects.

## Folder Role

`Metrics/` wraps reusable EB-JEPA metric implementations and adapts them to
Octave orchestration.

For AcVideoJepa, this includes prediction cost, anti-collapse regularization,
optional inverse dynamics loss, and the flat log dictionary consumed by
Lightning modules.

## File Roles

`ac_video_jepa_objective.py`

- defines `AcVideoJepaObjective`;
- evaluates an already-built metric set;
- passes flat metric values to an already-built weighted loss;
- returns metric logs plus loss logs for Lightning modules.

`prediction_metrics.py`

- defines one metric module per prediction-rollout comparison;
- separates autoregressive and parallel prediction losses.

`regularizer_metrics.py`

- defines one metric module per EB-JEPA elementary regularizer;
- wraps EB-JEPA standard-deviation, covariance, temporal-similarity, and inverse
  dynamics loss implementations;
- keeps each metric independently registered as an `nn.Module`.

`metric_set.py`

- defines generic metric-set utilities;
- defines `AcVideoJepaMetricSet`, which routes rollout outputs to metrics;
- converts metric outputs into a flat log dictionary.

`loss.py`

- defines `WeightedMetricLoss`;
- consumes a flat metric dictionary;
- computes a weighted scalar training loss.

`configs.py`

- stores plain objective defaults.

`factory.py`

- builds EB-JEPA projectors, inverse dynamics models, prediction costs, and
  regularizers from plain configs;
- builds metric modules, metric sets, and weighted loss objects;
- injects those already-built objects into `AcVideoJepaObjective`.

## Ownership Rules

Metrics code may:

- instantiate EB-JEPA loss and regularizer classes from plain configs;
- build objective-only helper modules such as projectors and inverse dynamics
  models;
- consume structured rollout outputs;
- aggregate flat metric values with a weighted loss;
- produce flat log dictionaries.

Metrics code must not:

- build encoders, action encoders, or predictors;
- implement rollout algorithms;
- parse dataloader batches beyond objective inputs it is explicitly given;
- configure optimizers or schedulers;
- resolve paths or read run YAML files.

## Extension Steps

1. Add plain objective defaults in `configs.py`.
2. Add objective construction in `factory.py`.
3. Wrap existing EB-JEPA metric implementations instead of recoding them.
4. Add focused unit tests for objective values, log keys, and semantic failures.
5. Update this README when objective behavior changes.
