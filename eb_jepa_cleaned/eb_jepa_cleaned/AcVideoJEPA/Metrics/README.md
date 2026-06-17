# AcVideoJEPA / Metrics

The JEPA objective for the experiment: the latent prediction loss and the
anti-collapse / auxiliary regularizers, expressed as metrics over a
`LatentRolloutOutput` and built on AIML's generic metric machinery (there is no
separate JEPA pillar — Decision 33).

## Files

- `primitives.py` — pure loss math re-implemented from EB-JEPA (Decision 30):
  `SquareLossSeq`, `HingeStdLoss`, `CovarianceLoss`, `TemporalSimilarityLoss`,
  `InverseDynamicsLoss`.
- `prediction.py` — `autoregressive_prediction_loss` / `parallel_prediction_loss`
  metrics (the core JEPA signal) + the sub-built `square_loss_seq` prediction
  cost.
- `regularizers.py` — `hinge_std`, `covariance`, `temporal_similarity`,
  `inverse_dynamics` metrics + the `RegularizerProjectionMixin` (5D reshaping) +
  the projector / inverse-dynamics field resolvers.
- `registry.py` — the local `PREDICTION_COST_REGISTRY` + `PREDICTION_COST_SUB_BUILD`.
- `factory.py` — importing it registers everything (metrics onto AIML's
  `METRIC_REGISTRY`, the cost onto the local registry).

## Contracts

- **Uniform metric signature** `forward(rollout_output, actions=None)`. Every
  metric receives the whole `LatentRolloutOutput` (and the raw actions, which
  only `inverse_dynamics` uses), so the module passes the same inputs to the
  whole set. Metrics route by registry **name** (not a `metric_type` field).
- **Regularizers act on `encoded_states [B, C, T, H, W]`**, reshaped by the
  mixin into sample matrices `[N, D]` (variance/covariance) or temporal layout
  `[T, B, D]` (temporal-similarity / inverse-dynamics), optionally after the
  projector.
- **Encoder-shape dependency.** The projector and inverse-dynamics model size
  depend on the encoder output shape, read from
  `runtime_context["encoder_shape"]` (`feature_dim`/`height`/`width`) by the
  field resolvers. The module injects it after probing the encoder; building a
  projector/IDM without it raises. (Routed via `runtime_context`, not a build
  kwarg, so it never leaks into a metric constructor.)
- **Prediction target alignment** differs by rollout mode, hence two prediction
  metrics; each checks the rollout's `unroll_mode` and raises on a mismatch.

## Tests

- `tests/unit/AcVideoJEPA/Metrics/` — primitive numerics, the prediction metrics
  over a `LatentRolloutOutput`, the regularizer reshaping + values, and building
  through the AIML metric factories (incl. the `encoder_shape` runtime-context
  path for a projector / inverse-dynamics build).
