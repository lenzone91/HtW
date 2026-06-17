# AcVideoJEPA

The concrete action-conditioned video JEPA experiment — the re-implementation of
EB-JEPA's `ac_video_jepa` example.

There is no separate JEPA pillar (Decision 33): JEPA is an encoder plus a loss
and regularizers over embeddings, so the JEPA objective is owned here, by the
experiment that uses it. This pillar learns a latent world model of an agent
navigating an environment from video, conditioned on the actions it takes. It is
the concrete experiment layer; it composes Workflow / AIML objects into a
runnable project and registers its concretes onto AIML registries.

## Owns

- the JEPA objective for this experiment: the latent prediction loss and the
  variance/covariance, temporal-similarity, and inverse-dynamics regularizers,
  as metrics over embeddings (built on AIML's `MetricSet` / `WeightedMetricLoss`);
- the video / frame-stack datasets (e.g. the two-rooms navigation dataset);
- the encoder model and action conditioning (action encoders, the
  action-conditioned predictor);
- the concrete encoder backbones (e.g. Impala / RNN);
- latent rollouts: autoregressive and parallel multi-step prediction in latent
  space;
- planning / evaluation trajectories over the learned latent dynamics;
- the JEPA Lightning module (the predict-in-latent-space training step);
- the experiment's Hydra config trees and run compositions;
- toy and smoke-level runnable examples.

## Must not own

- generic ML machinery — registries, builders, `BaseMetric`, `MetricSet`,
  `WeightedMetricLoss`, `BaseLightningModule`, optimizers/schedulers (belong in
  AIML).

## Dependencies

AcVideoJEPA may depend on Workflow and AIML. No pillar depends on AcVideoJEPA.

> Stub — migrated in Phase 4 (the concrete experiment + smoke test).
