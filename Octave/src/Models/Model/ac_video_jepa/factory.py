from copy import deepcopy

import torch
from torch import nn

from eb_jepa.architectures import (
    ImpalaEncoder,
    RNNPredictor,
)

from .configs import DEFAULT_AC_VIDEO_JEPA_COMPONENTS_CONFIG


class AcVideoJepaComponentsBuilder:
    """
    Build AcVideoJepa architecture components from a plain dictionary config.
    """

    def __init__(
        self,
        default_config: dict | None = None,
        strict: bool = True,
    ) -> None:
        self.default_config = deepcopy(
            default_config
            if default_config is not None
            else DEFAULT_AC_VIDEO_JEPA_COMPONENTS_CONFIG
        )
        self.strict = strict

    def __call__(
        self,
        config: dict,
        runtime_context: dict | None = None,
    ) -> dict:
        prepared_config = self.prepare_config(config)

        encoder = self.build_encoder(prepared_config["encoder"])
        encoder_shape = self.probe_encoder_shape(encoder, prepared_config["encoder"])
        action_encoder = self.build_action_encoder(prepared_config["action_encoder"])
        predictor = self.build_predictor(
            predictor_config=prepared_config["predictor"],
            encoder=encoder,
            encoder_dim=encoder_shape["feature_dim"],
        )

        return {
            "encoder": encoder,
            "action_encoder": action_encoder,
            "predictor": predictor,
            "encoder_shape": encoder_shape,
        }

    def prepare_config(self, config: dict) -> dict:
        if not isinstance(config, dict):
            raise TypeError(
                "AcVideoJepa components config must be a dictionary, "
                f"got {type(config).__name__}."
            )

        prepared_config = deepcopy(self.default_config)
        user_config = deepcopy(config)

        self.check_known_keys(user_config, prepared_config, section_name="components")

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


def build_ac_video_jepa_components(
    config: dict | None = None,
    runtime_context: dict | None = None,
    strict: bool = True,
) -> dict:
    builder = AcVideoJepaComponentsBuilder(strict=strict)
    return builder(
        config=config or DEFAULT_AC_VIDEO_JEPA_COMPONENTS_CONFIG,
        runtime_context=runtime_context,
    )
