from copy import deepcopy

import torch
from torch import nn

from eb_jepa.architectures import (
    ImpalaEncoder,
    InverseDynamicsModel,
    Projector,
    RNNPredictor,
)
from eb_jepa.losses import SquareLossSeq, VC_IDM_Sim_Regularizer

from .ac_video_jepa_model import AcVideoJepa
from .configs import DEFAULT_AC_VIDEO_JEPA_MODEL_CONFIG


class AcVideoJepaBuilder:
    """
    Build AcVideoJepa from a plain dictionary config.
    """

    def __init__(
        self,
        default_config: dict | None = None,
        strict: bool = True,
    ) -> None:
        self.default_config = deepcopy(
            default_config
            if default_config is not None
            else DEFAULT_AC_VIDEO_JEPA_MODEL_CONFIG
        )
        self.strict = strict

    def __call__(
        self,
        config: dict,
        runtime_context: dict | None = None,
    ) -> AcVideoJepa:
        prepared_config = self.prepare_config(config)

        encoder = self.build_encoder(prepared_config["encoder"])
        encoder_shape = self.probe_encoder_shape(encoder, prepared_config["encoder"])
        projector = self.build_projector(
            projector_config=prepared_config["projector"],
            encoder_dim=encoder_shape["feature_dim"],
        )
        action_encoder = self.build_action_encoder(prepared_config["action_encoder"])
        predictor = self.build_predictor(
            predictor_config=prepared_config["predictor"],
            encoder=encoder,
            encoder_dim=encoder_shape["feature_dim"],
        )
        idm = self.build_inverse_dynamics_model(
            idm_config=prepared_config["inverse_dynamics_model"],
            regularizer_config=prepared_config["regularizer"],
            projector=projector,
            encoder_shape=encoder_shape,
        )
        regularizer = self.build_regularizer(
            regularizer_config=prepared_config["regularizer"],
            projector=projector,
            idm=idm,
        )
        predcost = self.build_prediction_cost(
            prediction_cost_config=prepared_config["prediction_cost"],
            projector=projector,
        )

        return AcVideoJepa(
            encoder=encoder,
            aencoder=action_encoder,
            predictor=predictor,
            regularizer=regularizer,
            predcost=predcost,
        )

    def prepare_config(self, config: dict) -> dict:
        if not isinstance(config, dict):
            raise TypeError(
                "AcVideoJepa model config must be a dictionary, "
                f"got {type(config).__name__}."
            )

        prepared_config = deepcopy(self.default_config)
        user_config = deepcopy(config)

        self.check_known_keys(user_config, prepared_config, section_name="model")

        for key, value in user_config.items():
            if isinstance(value, dict) and isinstance(prepared_config.get(key), dict):
                self.check_known_keys(
                    value,
                    prepared_config[key],
                    section_name=key,
                )
                prepared_config[key].update(value)
            else:
                prepared_config[key] = value

        self.check_model_type(prepared_config)
        return prepared_config

    def build_encoder(self, encoder_config: dict) -> ImpalaEncoder:
        encoder_type = encoder_config["encoder_type"]
        if encoder_type != "impala":
            raise KeyError("Only 'impala' encoder_type is supported for AcVideoJepa.")

        kwargs = deepcopy(encoder_config)
        kwargs.pop("encoder_type")
        kwargs["stack_sizes"] = tuple(kwargs["stack_sizes"])
        kwargs["input_shape"] = tuple(kwargs["input_shape"])
        return ImpalaEncoder(**kwargs)

    def build_action_encoder(self, action_encoder_config: dict) -> nn.Module:
        action_encoder_type = action_encoder_config["action_encoder_type"]
        if action_encoder_type != "identity":
            raise KeyError(
                "Only 'identity' action_encoder_type is supported for AcVideoJepa."
            )

        return nn.Identity()

    def build_predictor(
        self,
        predictor_config: dict,
        encoder: ImpalaEncoder,
        encoder_dim: int,
    ) -> RNNPredictor:
        predictor_type = predictor_config["predictor_type"]
        if predictor_type != "rnn":
            raise KeyError("Only 'rnn' predictor_type is supported for AcVideoJepa.")

        hidden_size = predictor_config["hidden_size"] or encoder_dim
        final_ln = (
            encoder.final_ln
            if predictor_config["use_encoder_final_ln"]
            else nn.Identity()
        )

        return RNNPredictor(
            hidden_size=hidden_size,
            action_dim=predictor_config["action_dim"],
            num_layers=predictor_config["num_layers"],
            final_ln=final_ln,
        )

    def build_projector(
        self,
        projector_config: dict,
        encoder_dim: int,
    ) -> Projector | None:
        if not projector_config["enabled"]:
            return None

        mlp_spec = projector_config["mlp_spec"]
        if mlp_spec is None:
            hidden_dim = encoder_dim * projector_config["hidden_multiplier"]
            mlp_spec = f"{encoder_dim}-{hidden_dim}-{hidden_dim}"

        return Projector(mlp_spec)

    def build_inverse_dynamics_model(
        self,
        idm_config: dict,
        regularizer_config: dict,
        projector: Projector | None,
        encoder_shape: dict,
    ) -> InverseDynamicsModel | None:
        if not idm_config["enabled"]:
            return None

        if regularizer_config["idm_after_proj"] and projector is not None:
            state_channels = projector.out_dim
        else:
            state_channels = encoder_shape["feature_dim"]

        state_dim = (
            encoder_shape["height"]
            * encoder_shape["width"]
            * state_channels
        )

        return InverseDynamicsModel(
            state_dim=state_dim,
            hidden_dim=idm_config["hidden_dim"],
            action_dim=idm_config["action_dim"],
        )

    def build_regularizer(
        self,
        regularizer_config: dict,
        projector: Projector | None,
        idm: InverseDynamicsModel | None,
    ) -> VC_IDM_Sim_Regularizer:
        regularizer_type = regularizer_config["regularizer_type"]
        if regularizer_type != "vc_idm_sim":
            raise KeyError(
                "Only 'vc_idm_sim' regularizer_type is supported for AcVideoJepa."
            )

        kwargs = deepcopy(regularizer_config)
        kwargs.pop("regularizer_type")

        return VC_IDM_Sim_Regularizer(
            **kwargs,
            idm=idm,
            projector=projector,
        )

    def build_prediction_cost(
        self,
        prediction_cost_config: dict,
        projector: Projector | None,
    ) -> SquareLossSeq:
        prediction_cost_type = prediction_cost_config["prediction_cost_type"]
        if prediction_cost_type != "square_loss_seq":
            raise KeyError(
                "Only 'square_loss_seq' prediction_cost_type is supported "
                "for AcVideoJepa."
            )

        proj = projector if prediction_cost_config["use_projector"] else None
        return SquareLossSeq(proj=proj)

    def probe_encoder_shape(
        self,
        encoder: ImpalaEncoder,
        encoder_config: dict,
    ) -> dict:
        input_channels, height, width = encoder_config["input_shape"]

        with torch.no_grad():
            test_input = torch.zeros(1, input_channels, 1, height, width)
            test_output = encoder(test_input)

        _, feature_dim, _, output_height, output_width = test_output.shape

        return {
            "feature_dim": feature_dim,
            "height": output_height,
            "width": output_width,
        }

    def check_model_type(self, config: dict) -> None:
        if config["model_type"] != "ac_video_jepa":
            raise KeyError("Only 'ac_video_jepa' model_type is supported.")

    def check_known_keys(
        self,
        config: dict,
        default_config: dict,
        section_name: str,
    ) -> None:
        unknown_keys = set(config) - set(default_config)

        if not unknown_keys:
            return

        message = (
            f"Unknown AcVideoJepa {section_name} config keys: "
            f"{sorted(unknown_keys)}. "
            f"Allowed keys are: {sorted(default_config)}."
        )

        if self.strict:
            raise KeyError(message)


def build_ac_video_jepa(
    config: dict | None = None,
    runtime_context: dict | None = None,
    strict: bool = True,
) -> AcVideoJepa:
    builder = AcVideoJepaBuilder(strict=strict)
    return builder(
        config=config or DEFAULT_AC_VIDEO_JEPA_MODEL_CONFIG,
        runtime_context=runtime_context,
    )
