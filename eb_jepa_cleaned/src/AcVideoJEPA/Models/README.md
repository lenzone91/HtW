# AcVideoJEPA / Models

The experiment's neural-network architecture and Lightning runtime.

## Subfolders

- `Backbones/` — concrete building blocks: `ImpalaEncoder`, `RNNPredictor`
  (registered AIML models), `Projector`, `InverseDynamicsModel` (plain blocks).
- `Rollout/` — the latent rollout (autoregressive / parallel multi-step
  prediction) and its rollout-output structure. Same level as `Backbones/`.
- `Modules/` — the registered `AcVideoJepaModule`, the JEPA Lightning step that
  composes encoder + predictor + rollout + the JEPA objective (metric set +
  weighted loss).

## Registration

Backbone models and the Lightning module register onto AIML registries
(`MODEL_REGISTRY`, `LIGHTNING_MODULE_REGISTRY`) at import time (AcVideoJEPA ->
AIML). The JEPA objective metrics register onto the AIML metric registry from
`AcVideoJEPA/Metrics` (a sibling of `Models/`).
