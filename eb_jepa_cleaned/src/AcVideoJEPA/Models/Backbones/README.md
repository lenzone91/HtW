# AcVideoJEPA / Models / Backbones

The concrete neural-network building blocks of the experiment, re-implemented
from EB-JEPA (no `eb_jepa` dependency, Decision 30).

## Files

- `impala_encoder.py` — `ImpalaEncoder` (+ `ResnetBlock`, `ResnetStack`).
  Encodes a clip `[B, C, T, H, W]` frame-by-frame to latents `[B, D, T, 1, 1]`.
  Registered model `impala_encoder`. Exposes `final_ln` so a predictor can reuse
  it.
- `rnn_predictor.py` — `RNNPredictor`. Single-step GRU advancing a latent state
  given an action (`state [B, D, 1, 1, 1]`, `action [B, A, 1]`). Registered
  model `rnn_predictor`. Flags `is_rnn=True`, `context_length=0` for the rollout.
- `projector.py` — `Projector`, an expander MLP from a spec string. Plain block.
- `inverse_dynamics.py` — `InverseDynamicsModel`, predicts the action from two
  consecutive states; `init_module_weights` helper. Plain block.

## Registration

`ImpalaEncoder` and `RNNPredictor` register onto
`AIML.Models.Models.registry.MODEL_REGISTRY` at import time and are built with
`build_model(config, model_name=...)`. `Projector` / `InverseDynamicsModel` are
**not** registered models — they are sub-objects the regularizer metrics build
via field resolvers (they depend on the encoder feature dimension, known only at
module-build time).

## Deferred wiring

Two cross-object couplings are resolved later, when the Lightning module is
assembled (it has both the encoder and the predictor config):

- the predictor's `hidden_size` defaulting to the encoder feature dimension;
- the predictor optionally reusing the encoder's `final_ln`
  (`use_encoder_final_ln`).

These are module-build field resolutions, not backbone responsibilities.

## Faithfulness note

Clean re-implementations of the EB-JEPA modules. One correctness fix:
`ImpalaEncoder` wires each ResNet stack's input channels to the previous stack's
output channels (`stack_sizes[i] * width`); identical to EB-JEPA at the default
`width=1`, correct for `width != 1`.

## Tests

- `tests/unit/AcVideoJEPA/Models/Backbones/` — shape probes (encoder
  `[B,C,T,H,W] -> [B,D,T,1,1]`, one predictor step, projector, inverse
  dynamics) and building the registered models through the AIML model factory.
