# AcVideoJEPA / Models / Modules

The registered `AcVideoJepaModule` — the JEPA Lightning step over the latent
rollout. The keystone that composes the experiment.

## Files

- `ac_video_jepa_module.py` — `AcVideoJepaModule(BaseLightningModule)` + its
  build resolvers and `DEFAULT_AC_VIDEO_JEPA_MODULE_CONFIG`. Registered
  `ac_video_jepa` on the AIML Lightning-module registry.
- `factory.py` — imports the module (and its build dependencies: backbones,
  rollout, objective metrics) for their registration side effects.

## Runtime interface

The module holds `encoder` / `predictor` / `action_encoder` in `self.models` and
exposes `encode` / `encode_actions` / `predict` so the latent rollout can drive
it. The step is:

    batch {states, actions} -> rollout(self, states, actions) -> LatentRolloutOutput
      -> metric set (each metric gets (rollout_output, actions))
      -> WeightedMetricLoss -> log

## Build wiring (field-resolution chain)

Built through the generic `build_lightning_module`. The resolvers run in order:

1. `resolve_models` — builds the `impala_encoder`, probes its output shape,
   builds the `rnn_predictor` (its `hidden_size` defaults to the encoder feature
   dim; optionally reuses the encoder `final_ln`), and the identity action
   encoder. Stashes `encoder_shape` on the config.
2. `resolve_metric_set` — builds the objective metric set with `encoder_shape`
   injected into `runtime_context` (the projector / inverse-dynamics field
   resolvers read it there).
3. `resolve_rollout` / `resolve_loss` — build the rollout and the weighted loss.

## Optimization

`configure_optimizers` builds a **single** optimizer over `self.parameters()` —
covering the encoder, predictor, and any trainable objective parameters
(inverse-dynamics model / projectors) held by the metric set, which the base
per-model default would miss.

## Tests

- `tests/unit/AcVideoJEPA/Models/Modules/` — building through
  `build_lightning_module`, one `compute_step`, `configure_optimizers` covering
  metric params, and a one-step `trainer.fit` on a tiny config.
