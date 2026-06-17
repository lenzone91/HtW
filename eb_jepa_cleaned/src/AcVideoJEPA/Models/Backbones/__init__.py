"""
AcVideoJEPA backbones: the concrete neural-network building blocks of the
experiment (encoder, predictor, projector, inverse-dynamics model). Importing
this subpackage registers the backbone models onto the AIML model registry.
"""

from . import factory  # noqa: F401  (registration side effect)
from .impala_encoder import ImpalaEncoder
from .inverse_dynamics import InverseDynamicsModel
from .projector import Projector
from .rnn_predictor import RNNPredictor

__all__ = [
    "ImpalaEncoder",
    "RNNPredictor",
    "Projector",
    "InverseDynamicsModel",
]
