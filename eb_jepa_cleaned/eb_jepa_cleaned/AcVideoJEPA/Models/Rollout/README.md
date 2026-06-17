# AcVideoJEPA / Models / Rollout

The loss-free latent rollout: encode a clip, then roll the predictor forward in
latent space. It computes no loss — the JEPA objective metrics consume its
output.

## Files

- `output.py` — `LatentRolloutOutput`, the frozen dataclass the metrics consume
  (`encoded_states`, `predicted_states`, `actions_encoded`, `nsteps`,
  `unroll_mode`, `effective_ctxt_window`, `predicted_steps`).
- `registry.py` — `ROLLOUT_REGISTRY` + `ROLLOUT_BUILDER` (local registry, routed
  by `rollout_type`). Rollout is experiment-specific, so the registry is local to
  AcVideoJEPA, not in AIML.
- `latent_rollout.py` — `LatentRollout` (registered `latent`). Modes:
  - `parallel`: predict every step from the encoded context each iteration;
  - `autoregressive`: feed predictions back, one step at a time.
- `factory.py` — `build_rollout`; imports the object module for registration.

## Contract

`LatentRollout(jepa, observations, actions)` requires the `jepa` object to expose
`encode`, `encode_actions`, `predict`, and a `predictor` (its `is_rnn` /
`context_length` attributes select the effective context window: 1 for an RNN
predictor, else `ctxt_window_time`). The Lightning module is that `jepa` object.

## Tests

- `tests/unit/AcVideoJEPA/Models/Rollout/` — autoregressive rollout with a real
  encoder + RNN predictor, parallel rollout with a stub predictor, config
  validation, and the output structure / mode / context window.
