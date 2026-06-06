"""Factories for common EB-JEPA world-model components."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from Enzo.jepa_templates.pathing import ensure_eb_jepa_importable


@dataclass
class ACJEPAComponents:
    """Objects created for an action-conditioned JEPA model."""

    jepa: Any
    encoder: Any
    predictor: Any
    regularizer: Any
    pred_loss: Any
    projector: Any | None
    idm: Any
    encoder_output_shape: tuple[int, ...]


def build_impala_rnn_ac_jepa(
    cfg: Any,
    img_size: int,
    action_dim: int = 2,
    device: Any | None = None,
) -> ACJEPAComponents:
    """Build the Impala + GRU action-conditioned JEPA recipe.

    This mirrors the strongest EB-JEPA example while keeping the model creation
    isolated from data loading and training orchestration.
    """
    ensure_eb_jepa_importable()

    import torch
    import torch.nn as nn

    from eb_jepa.architectures import (
        ImpalaEncoder,
        InverseDynamicsModel,
        Projector,
        RNNPredictor,
    )
    from eb_jepa.jepa import JEPA
    from eb_jepa.losses import SquareLossSeq, VC_IDM_Sim_Regularizer

    dobs = _cfg_get(cfg, "model.dobs", 2)
    henc = _cfg_get(cfg, "model.henc", 32)
    dstc = _cfg_get(cfg, "model.dstc", 32)
    mlp_output_dim = _cfg_get(cfg, "model.mlp_output_dim", 512)

    encoder = ImpalaEncoder(
        width=1,
        stack_sizes=(16, henc, dstc),
        num_blocks=_cfg_get(cfg, "model.num_blocks", 2),
        dropout_rate=_cfg_get(cfg, "model.dropout_rate", None),
        layer_norm=_cfg_get(cfg, "model.layer_norm", False),
        input_channels=dobs,
        final_ln=_cfg_get(cfg, "model.final_ln", True),
        mlp_output_dim=mlp_output_dim,
        input_shape=(dobs, img_size, img_size),
    )

    test_input = torch.rand((1, dobs, 1, img_size, img_size))
    test_output = encoder(test_input)
    _, feature_dim, _, h, w = test_output.shape

    predictor = RNNPredictor(
        hidden_size=encoder.mlp_output_dim,
        action_dim=action_dim,
        num_layers=_cfg_get(cfg, "model.predictor_num_layers", 1),
        final_ln=encoder.final_ln if hasattr(encoder, "final_ln") else nn.Identity(),
    )

    projector = None
    if _cfg_get(cfg, "model.regularizer.use_proj", False):
        projector = Projector(
            f"{encoder.mlp_output_dim}-{encoder.mlp_output_dim * 4}-{encoder.mlp_output_dim * 4}"
        )

    idm_after_proj = _cfg_get(cfg, "model.regularizer.idm_after_proj", False)
    idm_state_dim = h * w * (
        projector.out_dim if projector is not None and idm_after_proj else feature_dim
    )
    idm = InverseDynamicsModel(
        state_dim=idm_state_dim,
        hidden_dim=_cfg_get(cfg, "model.regularizer.idm_hidden_dim", 256),
        action_dim=action_dim,
    )

    regularizer = VC_IDM_Sim_Regularizer(
        cov_coeff=_cfg_get(cfg, "model.regularizer.cov_coeff", 8.0),
        std_coeff=_cfg_get(cfg, "model.regularizer.std_coeff", 16.0),
        sim_coeff_t=_cfg_get(cfg, "model.regularizer.sim_coeff_t", 12.0),
        idm_coeff=_cfg_get(cfg, "model.regularizer.idm_coeff", 1.0),
        idm=idm,
        first_t_only=_cfg_get(cfg, "model.regularizer.first_t_only", False),
        projector=projector,
        spatial_as_samples=_cfg_get(cfg, "model.regularizer.spatial_as_samples", False),
        idm_after_proj=idm_after_proj,
        sim_t_after_proj=_cfg_get(cfg, "model.regularizer.sim_t_after_proj", False),
    )

    pred_loss = SquareLossSeq()
    jepa = JEPA(
        encoder=encoder,
        aencoder=nn.Identity(),
        predictor=predictor,
        regularizer=regularizer,
        predcost=pred_loss,
    )

    if device is not None:
        jepa = jepa.to(device)

    return ACJEPAComponents(
        jepa=jepa,
        encoder=encoder,
        predictor=predictor,
        regularizer=regularizer,
        pred_loss=pred_loss,
        projector=projector,
        idm=idm,
        encoder_output_shape=tuple(test_output.shape),
    )


def _cfg_get(cfg: Any, dotted_key: str, default: Any = None) -> Any:
    current = cfg
    for key in dotted_key.split("."):
        if current is None:
            return default
        if isinstance(current, dict):
            current = current.get(key, default)
        else:
            current = getattr(current, key, default)
    return current

