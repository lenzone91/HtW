import torch

from torch import nn
from eb_jepa.architectures import ImpalaEncoder, RNNPredictor

from .configs import DEFAULT_AC_VIDEO_JEPA_COMPONENTS_CONFIG
from .registry import AC_VIDEO_JEPA_COMPONENTS_REGISTRY


@AC_VIDEO_JEPA_COMPONENTS_REGISTRY.register_class(
    name="ac_video_jepa",
    default_config=DEFAULT_AC_VIDEO_JEPA_COMPONENTS_CONFIG,
    type_field="model_type",
)
class AcVideoJepaComponents:
    """
    Architecture-only AcVideoJepa component bundle.

    This object owns construction of:
    - encoder;
    - action encoder;
    - predictor;
    - encoder shape metadata.

    It does not own rollout, losses, metrics, optimizers, schedulers, or
    Lightning step orchestration.
    """

    def __init__(
        self,
        encoder: dict,
        action_encoder: dict,
        predictor: dict,
    ) -> None:
        self.encoder = self.build_encoder(encoder)
        self.encoder_shape = self.probe_encoder_shape(
            encoder=self.encoder,
            encoder_config=encoder,
        )
        self.action_encoder = self.build_action_encoder(action_encoder)
        self.predictor = self.build_predictor(
            predictor_config=predictor,
            encoder=self.encoder,
            encoder_dim=self.encoder_shape["feature_dim"],
        )

    def as_dict(self) -> dict:
        return {
            "encoder": self.encoder,
            "action_encoder": self.action_encoder,
            "predictor": self.predictor,
            "encoder_shape": self.encoder_shape,
        }

    def build_encoder(self, encoder_config: dict) -> ImpalaEncoder:
        encoder_type = encoder_config["encoder_type"]

        if encoder_type != "impala":
            raise KeyError("Only 'impala' encoder_type is supported.")

        kwargs = dict(encoder_config)
        kwargs.pop("encoder_type")
        kwargs["stack_sizes"] = tuple(kwargs["stack_sizes"])
        kwargs["input_shape"] = tuple(kwargs["input_shape"])

        return ImpalaEncoder(**kwargs)

    def build_action_encoder(self, action_encoder_config: dict) -> nn.Module:
        action_encoder_type = action_encoder_config["action_encoder_type"]

        if action_encoder_type != "identity":
            raise KeyError("Only 'identity' action_encoder_type is supported.")

        return nn.Identity()

    def build_predictor(
        self,
        predictor_config: dict,
        encoder: ImpalaEncoder,
        encoder_dim: int,
    ) -> RNNPredictor:
        predictor_type = predictor_config["predictor_type"]

        if predictor_type != "rnn":
            raise KeyError("Only 'rnn' predictor_type is supported.")

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