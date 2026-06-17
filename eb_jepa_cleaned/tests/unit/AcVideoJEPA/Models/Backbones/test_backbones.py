"""
AcVideoJEPA backbones: shape contracts and registration onto the AIML model
registry. Small dims keep the encoder probe fast.
"""

import torch

# Importing the subpackage registers the backbone models onto MODEL_REGISTRY.
import eb_jepa_cleaned.AcVideoJEPA.Models.Backbones  # noqa: F401
from eb_jepa_cleaned.AcVideoJEPA.Models.Backbones.impala_encoder import ImpalaEncoder
from eb_jepa_cleaned.AcVideoJEPA.Models.Backbones.inverse_dynamics import (
    InverseDynamicsModel,
)
from eb_jepa_cleaned.AcVideoJEPA.Models.Backbones.projector import Projector
from eb_jepa_cleaned.AcVideoJEPA.Models.Backbones.rnn_predictor import RNNPredictor
from eb_jepa_cleaned.AIML.Models.Models.factory import build_model
from eb_jepa_cleaned.AIML.Models.Models.registry import MODEL_REGISTRY

SMALL_ENCODER = {
    "input_shape": [2, 16, 16],
    "input_channels": 2,
    "stack_sizes": [4, 8],
    "num_blocks": 1,
    "mlp_output_dim": 8,
}


#############################################
# Registration
#############################################


def test_backbones_registered():
    assert MODEL_REGISTRY.has("impala_encoder")
    assert MODEL_REGISTRY.has("rnn_predictor")


#############################################
# ImpalaEncoder
#############################################


def test_impala_encoder_shape_contract():
    encoder = ImpalaEncoder(**SMALL_ENCODER)
    clip = torch.zeros(2, 2, 3, 16, 16)  # [B, C, T, H, W]
    out = encoder(clip)
    assert out.shape == (2, 8, 3, 1, 1)  # [B, D, T, 1, 1]


def test_impala_encoder_builds_through_factory():
    encoder = build_model(SMALL_ENCODER, model_name="impala_encoder")
    assert isinstance(encoder, ImpalaEncoder)
    assert encoder.mlp_output_dim == 8


#############################################
# RNNPredictor
#############################################


def test_rnn_predictor_single_step_shape():
    predictor = RNNPredictor(hidden_size=8, action_dim=2, num_layers=1)
    state = torch.zeros(4, 8, 1, 1, 1)
    action = torch.zeros(4, 2, 1)
    out = predictor(state, action)
    assert out.shape == (4, 8, 1, 1, 1)
    assert predictor.is_rnn is True
    assert predictor.context_length == 0


def test_rnn_predictor_builds_through_factory():
    predictor = build_model(
        {"hidden_size": 8, "action_dim": 2, "num_layers": 1},
        model_name="rnn_predictor",
    )
    assert isinstance(predictor, RNNPredictor)


#############################################
# Projector / InverseDynamicsModel
#############################################


def test_projector_shape_and_out_dim():
    projector = Projector("8-16-4")
    out = projector(torch.randn(5, 8))
    assert out.shape == (5, 4)
    assert projector.out_dim == 4


def test_inverse_dynamics_predicts_action_shape():
    idm = InverseDynamicsModel(state_dim=8, hidden_dim=16, action_dim=2)
    state_t = torch.randn(4, 8)
    state_t1 = torch.randn(4, 8)
    assert idm(state_t, state_t1).shape == (4, 2)
