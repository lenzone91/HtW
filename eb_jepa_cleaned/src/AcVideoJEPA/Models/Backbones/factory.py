"""
AcVideoJEPA backbones factory.

Importing this module registers the backbone models (`impala_encoder`,
`rnn_predictor`) onto the AIML model registry (the registration side effect).
They are then built through the ordinary AIML model factory.

`Projector` and `InverseDynamicsModel` are plain building blocks, not registered
models: the regularizer metrics construct them via their own field resolvers.
"""

from . import impala_encoder  # noqa: F401  (registers impala_encoder)
from . import rnn_predictor  # noqa: F401  (registers rnn_predictor)

# Backbone models registered onto the AIML MODEL_REGISTRY by importing this.
REGISTERED_MODELS = (
    "impala_encoder",
    "rnn_predictor",
)
