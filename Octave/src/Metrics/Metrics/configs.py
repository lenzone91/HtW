DEFAULT_SQUARE_LOSS_SEQ_CONFIG = {
    "prediction_cost_type": "square_loss_seq",
    "use_projector": False,
    "proj": None,
}


DEFAULT_AUTOREGRESSIVE_PREDICTION_LOSS_METRIC_CONFIG = {
    "metric_type": "autoregressive_prediction_loss",
    "prediction_cost": dict(DEFAULT_SQUARE_LOSS_SEQ_CONFIG),
}


DEFAULT_PARALLEL_PREDICTION_LOSS_METRIC_CONFIG = {
    "metric_type": "parallel_prediction_loss",
    "prediction_cost": dict(DEFAULT_SQUARE_LOSS_SEQ_CONFIG),
}


DEFAULT_PROJECTOR_CONFIG = {
    "enabled": False,
    "mlp_spec": None,
    "hidden_multiplier": 4,
}


DEFAULT_INVERSE_DYNAMICS_MODEL_CONFIG = {
    "enabled": True,
    "hidden_dim": 256,
    "action_dim": 2,
}


DEFAULT_HINGE_STD_LOSS_METRIC_CONFIG = {
    "metric_type": "hinge_std",
    "projector": dict(DEFAULT_PROJECTOR_CONFIG),
    "std_margin": 1.0,
    "first_t_only": False,
    "spatial_as_samples": False,
}


DEFAULT_COVARIANCE_LOSS_METRIC_CONFIG = {
    "metric_type": "covariance",
    "projector": dict(DEFAULT_PROJECTOR_CONFIG),
    "first_t_only": False,
    "spatial_as_samples": False,
}


DEFAULT_TEMPORAL_SIMILARITY_LOSS_METRIC_CONFIG = {
    "metric_type": "temporal_similarity",
    "projector": dict(DEFAULT_PROJECTOR_CONFIG),
    "after_projection": False,
}


DEFAULT_INVERSE_DYNAMICS_LOSS_METRIC_CONFIG = {
    "metric_type": "inverse_dynamics",
    "projector": dict(DEFAULT_PROJECTOR_CONFIG),
    "inverse_dynamics_model": dict(DEFAULT_INVERSE_DYNAMICS_MODEL_CONFIG),
    "after_projection": False,
}


DEFAULT_METRIC_CONFIGS = {
    "prediction_loss": dict(DEFAULT_AUTOREGRESSIVE_PREDICTION_LOSS_METRIC_CONFIG),
    "std_loss": dict(DEFAULT_HINGE_STD_LOSS_METRIC_CONFIG),
    "cov_loss": dict(DEFAULT_COVARIANCE_LOSS_METRIC_CONFIG),
    "sim_loss_t": dict(DEFAULT_TEMPORAL_SIMILARITY_LOSS_METRIC_CONFIG),
    "idm_loss": dict(DEFAULT_INVERSE_DYNAMICS_LOSS_METRIC_CONFIG),
}