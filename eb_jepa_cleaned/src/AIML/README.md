# AIML

Generic, domain-agnostic AI/ML pipeline machinery. It owns the reusable ML
abstractions; it must not contain experiment-specific logic (including the JEPA
objective, which is an encoder + metrics owned by the experiment, Decision 33).

## Subsystems

- `Data/` — datasets, batch transforms (`DataAugmentation` / `DataAdaptation`
  over `BatchTransform`), collators, Lightning DataModules.
- `Metrics/` — metrics (`BaseMetric`), metric sets (`MetricSet` /
  `LoggableMetricSet`), and `WeightedMetricLoss`.
- `Models/` — model registry/factory, `BaseLightningModule`, and weight loading.
- `Training/` — optimizers, schedulers, loggers, checkpoints, early stoppings
  (thin torch/Lightning wrappers).
- `Execution/` — run composition: datamodule + module + Trainer.

## Dependencies

AIML may depend on Workflow. It must not depend on AcVideoJEPA.

## Machinery vs concretes

AIML holds the machinery (bases, registries, builders, factories) plus the few
generic concretes (`DefaultDataModule`, `MetricSet`/`LoggableMetricSet`,
`WeightedMetricLoss`, torch optimizers/schedulers, Lightning callbacks,
`BaseLightningModule`). All experiment concretes — the JEPA objective metrics
(latent prediction loss + anti-collapse regularizers), the encoder backbones and
action conditioning, video datasets, rollouts, and the JEPA Lightning module —
belong to AcVideoJEPA (Phase 4) and register onto AIML's registries at import
time (Decisions 26/33). AIML registries that ship empty of concretes are tested
with dummy generic objects (Decision 26).

## Deferred

Runtime-context path resolution (datasets, checkpoints, loggers, loading) and the
Setup-coupled run orchestration (`run_training`/`run_evaluation`, reports,
snapshots) are deferred to the Setup migration (Decision 22). Sweeps (W&B) are
deferred (Decision 26).

## Flow

    Hydra config -> plain dict -> factories -> objects
      data: dataset -> collator (augmentation/adaptation) -> DataModule
      model: nn.Module ; module: BaseLightningModule (built model/metrics/loss)
      training: optimizers/schedulers/loggers/callbacks
      execution: build_training_objects -> trainer.fit
