"""
AcVideoJEPA Lightning module — the JEPA training step over the latent rollout.

This is the keystone that composes the experiment: it holds the encoder,
predictor, and action encoder (as `models`), exposes `encode`/`encode_actions`/
`predict` so the rollout can drive it, and runs the step

    batch -> latent rollout -> JEPA objective metric set -> weighted loss -> log

Build wiring (cross-object, done as an ordered field-resolution chain so the
module builds through the generic `build_lightning_module`):

1. `resolve_models` builds the encoder, probes its output shape, builds the
   predictor (its `hidden_size` defaults to the encoder feature dim; it may reuse
   the encoder's `final_ln`), and the action encoder. The probed `encoder_shape`
   is stashed on the config.
2. `resolve_metric_set` builds the objective metric set with `encoder_shape`
   injected into `runtime_context` (the projector / inverse-dynamics field
   resolvers read it there).
3. `resolve_rollout` / `resolve_loss` build the rollout and the weighted loss.

Encoder/predictor are fixed to the registered `impala_encoder` / `rnn_predictor`
(the only architectures `ac_video_jepa` uses).
"""

import torch
from torch import nn

from ....AIML.Metrics.Loss.factory import build_loss
from ....AIML.Metrics.MetricSets.factory import build_metric_set
from ....AIML.Models.Models.factory import build_model
from ....AIML.Models.Modules.base import BaseLightningModule
from ....AIML.Models.Modules.registry import LIGHTNING_MODULE_REGISTRY
from ....AIML.Training.Optimizers.factory import build_optimizers
from ....AIML.Training.Schedulers.factory import build_schedulers
from ....Workflow.Factory.registry import FieldResolution
from ..Rollout.factory import build_rollout


#############################################
# Build helpers (cross-object wiring)
#############################################


def probe_encoder_shape(encoder: nn.Module) -> dict:
    """Probe the encoder output shape with a one-frame dummy clip."""
    input_channels, height, width = encoder.input_shape
    with torch.no_grad():
        dummy = torch.zeros(1, input_channels, 1, height, width)
        output = encoder(dummy)
    _, feature_dim, _, out_height, out_width = output.shape
    return {"feature_dim": feature_dim, "height": out_height, "width": out_width}


def build_predictor(predictor_config, encoder, encoder_shape, runtime_context=None):
    """Build the predictor, coupling it to the encoder where requested."""
    predictor_config = dict(predictor_config)
    use_encoder_final_ln = predictor_config.pop("use_encoder_final_ln", False)
    if not predictor_config.get("hidden_size"):
        predictor_config["hidden_size"] = encoder_shape["feature_dim"]

    predictor = build_model(
        predictor_config, model_name="rnn_predictor", runtime_context=runtime_context
    )
    if use_encoder_final_ln:
        predictor.final_ln = encoder.final_ln
    return predictor


def build_action_encoder(action_encoder_config) -> nn.Module:
    action_encoder_type = dict(action_encoder_config).get(
        "action_encoder_type", "identity"
    )
    if action_encoder_type != "identity":
        raise ValueError(
            f"Unsupported action_encoder_type '{action_encoder_type}'; "
            "only 'identity' is supported."
        )
    return nn.Identity()


#############################################
# Field resolvers
#############################################


def resolve_models(config, runtime_context=None, **kwargs) -> dict:
    encoder = build_model(
        config["encoder"], model_name="impala_encoder", runtime_context=runtime_context
    )
    encoder_shape = probe_encoder_shape(encoder)
    predictor = build_predictor(
        config["predictor"], encoder, encoder_shape, runtime_context
    )
    action_encoder = build_action_encoder(config["action_encoder"])

    # Stash the probed shape for the metric-set resolver (runs next).
    config["encoder_shape"] = encoder_shape
    return {"encoder": encoder, "predictor": predictor, "action_encoder": action_encoder}


def resolve_metric_set(config, runtime_context=None, **kwargs):
    metric_runtime_context = {
        **(runtime_context or {}),
        "encoder_shape": config["encoder_shape"],
    }
    return build_metric_set(config["metrics"], runtime_context=metric_runtime_context)


def resolve_rollout(config, runtime_context=None, **kwargs):
    return build_rollout(config["rollout"], runtime_context=runtime_context)


def resolve_loss(config, runtime_context=None, **kwargs):
    return build_loss(config["loss"], runtime_context=runtime_context)


MODELS_FIELD = FieldResolution(
    target_key="models",
    resolver=resolve_models,
    remove_source_keys=("encoder", "predictor", "action_encoder"),
)
METRIC_SET_FIELD = FieldResolution(
    target_key="metric_set",
    resolver=resolve_metric_set,
    remove_source_keys=("metrics",),
)
ROLLOUT_FIELD = FieldResolution(target_key="rollout", resolver=resolve_rollout)
LOSS_FIELD = FieldResolution(target_key="loss", resolver=resolve_loss)


#############################################
# Default config
#############################################

# Canonical default (allow-list + a buildable default experiment). Inner configs
# are complete because the builder does not merge defaults.
DEFAULT_AC_VIDEO_JEPA_MODULE_CONFIG = {
    "module_type": "ac_video_jepa",
    "encoder": {
        "stack_sizes": [16, 32, 32],
        "num_blocks": 2,
        "input_channels": 2,
        "mlp_output_dim": 512,
        "input_shape": [2, 65, 65],
    },
    "predictor": {
        "hidden_size": None,  # -> encoder feature dim
        "action_dim": 2,
        "num_layers": 1,
        "use_encoder_final_ln": True,
    },
    "action_encoder": {"action_encoder_type": "identity"},
    "rollout": {
        "rollout_type": "latent",
        "nsteps": 1,
        "unroll_mode": "autoregressive",
        "ctxt_window_time": 1,
        "return_all_steps": False,
    },
    "metrics": {
        "set_type": "metric",
        "metrics": {
            "autoregressive_prediction_loss": {
                "prediction_cost": {
                    "prediction_cost_type": "square_loss_seq",
                    "use_projector": False,
                    "proj": None,
                }
            },
            "hinge_std": {
                "projector": {"enabled": False, "mlp_spec": None, "hidden_multiplier": 4},
                "std_margin": 1.0,
                "first_t_only": False,
                "spatial_as_samples": False,
            },
            "covariance": {
                "projector": {"enabled": False, "mlp_spec": None, "hidden_multiplier": 4},
                "first_t_only": False,
                "spatial_as_samples": False,
            },
        },
    },
    "loss": {
        "loss_type": "weighted_metric",
        "metric_weights": {
            "autoregressive_prediction_loss": 1.0,
            "hinge_std": 1.0,
            "covariance": 1.0,
        },
    },
    "encoder_shape": None,  # filled by resolve_models
    "optimizer_configs": {"optimizer": {"optimizer_type": "adam", "lr": 1e-3}},
    "scheduler_configs": {},
}


@LIGHTNING_MODULE_REGISTRY.register_class(
    name="ac_video_jepa",
    default_config=DEFAULT_AC_VIDEO_JEPA_MODULE_CONFIG,
    type_field="module_type",
    field_resolutions=(MODELS_FIELD, METRIC_SET_FIELD, ROLLOUT_FIELD, LOSS_FIELD),
)
class AcVideoJepaModule(BaseLightningModule):
    """Lightning orchestration for AcVideoJEPA."""

    required_batch_keys = ("states", "actions")

    def __init__(
        self,
        models: dict,
        rollout,
        metric_set: nn.Module,
        loss: nn.Module,
        optimizer_configs: dict,
        scheduler_configs: dict,
        encoder_shape: dict | None = None,
    ) -> None:
        super().__init__(
            models=models,
            optimizer_configs=optimizer_configs,
            scheduler_configs=scheduler_configs,
        )
        self.rollout = rollout
        self.metric_set = metric_set
        self.loss = loss
        self.encoder_shape = dict(encoder_shape) if encoder_shape else None

    #############################################
    # JEPA runtime interface (driven by the rollout)
    #############################################

    @property
    def predictor(self) -> nn.Module:
        # The rollout reads `jepa.predictor` (its `is_rnn` / `context_length`)
        # to choose the effective context window.
        return self.models["predictor"]

    def encode(self, observations: torch.Tensor) -> torch.Tensor:
        return self.models["encoder"](observations)

    def encode_actions(self, actions: torch.Tensor | None) -> torch.Tensor | None:
        if actions is None:
            return None
        return self.models["action_encoder"](actions)

    def predict(self, states: torch.Tensor, actions: torch.Tensor | None) -> torch.Tensor:
        return self.models["predictor"](states, actions)

    def forward(self, observations: torch.Tensor, actions: torch.Tensor):
        return self.rollout(self, observations, actions)

    #############################################
    # Step
    #############################################

    def compute_step(self, batch: dict):
        self.check_batch(batch)
        rollout_output = self.rollout(self, batch["states"], batch["actions"])
        metrics_inputs = {
            name: (rollout_output, batch["actions"]) for name in self.metric_set.metrics
        }
        metric_values = self.metric_set(metrics_inputs)
        total_loss, loss_logs = self.loss(metric_values)
        return total_loss, metric_values, loss_logs

    def training_step(self, batch, batch_idx: int) -> torch.Tensor:
        total_loss, metric_values, loss_logs = self.compute_step(batch)
        self.log_step_dict(
            "train", metric_values, loss_logs,
            prog_bar=True, on_step=True, on_epoch=True, logger=True,
        )
        return total_loss

    def validation_step(self, batch, batch_idx: int) -> torch.Tensor:
        total_loss, metric_values, loss_logs = self.compute_step(batch)
        self.log_step_dict(
            "val", metric_values, loss_logs,
            prog_bar=False, on_step=False, on_epoch=True, logger=True,
        )
        return total_loss

    def test_step(self, batch, batch_idx: int) -> torch.Tensor:
        total_loss, metric_values, loss_logs = self.compute_step(batch)
        self.log_step_dict(
            "test", metric_values, loss_logs,
            prog_bar=False, on_step=False, on_epoch=True, logger=True,
        )
        return total_loss

    #############################################
    # Optimization
    #############################################

    def configure_optimizers(self):
        # One optimizer over ALL trainable parameters: encoder, predictor, and any
        # trainable objective params (inverse-dynamics model, projectors) held by
        # the metric set. The per-model base default would miss the metric params.
        if len(self.optimizer_configs) != 1:
            raise ValueError(
                "AcVideoJepaModule expects exactly one optimizer config (it "
                "optimizes all parameters jointly)."
            )

        parameter_groups = {name: self.parameters() for name in self.optimizer_configs}
        optimizers = build_optimizers(parameter_groups, self.optimizer_configs)
        schedulers = build_schedulers(optimizers, self.scheduler_configs)

        optimizers = list(optimizers.values())
        schedulers = list(schedulers.values())
        if not schedulers:
            return optimizers
        return optimizers, schedulers

    #############################################
    # Validation
    #############################################

    def check_batch(self, batch: dict) -> None:
        if not isinstance(batch, dict):
            raise TypeError(
                f"AcVideoJepaModule expected a dict batch, got {type(batch).__name__}."
            )
        missing = [k for k in self.required_batch_keys if k not in batch]
        if missing:
            raise KeyError(f"AcVideoJepaModule batch is missing keys: {missing}.")
        for key in self.required_batch_keys:
            if not isinstance(batch[key], torch.Tensor):
                raise TypeError(
                    f"AcVideoJepaModule batch key '{key}' must be a torch.Tensor, "
                    f"got {type(batch[key]).__name__}."
                )
