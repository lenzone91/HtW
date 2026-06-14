DEFAULT_AC_VIDEO_JEPA_OBJECTIVE_CONFIG = {
    "objective_type": "ac_video_jepa",
    "metric_set": {
        "set_type": "ac_video_jepa",
        "strict": True,
        "metric_to_input_names": None,
        "metrics": {
            "prediction_loss": {
                "metric_type": "autoregressive_prediction_loss",
                "prediction_cost": {
                    "prediction_cost_type": "square_loss_seq",
                    "use_projector": False,
                },
            },
            "std_loss": {
                "metric_type": "hinge_std",
                "projector": {
                    "enabled": False,
                    "mlp_spec": None,
                    "hidden_multiplier": 4,
                },
                "std_margin": 1.0,
                "first_t_only": False,
                "spatial_as_samples": False,
            },
            "cov_loss": {
                "metric_type": "covariance",
                "projector": {
                    "enabled": False,
                    "mlp_spec": None,
                    "hidden_multiplier": 4,
                },
                "first_t_only": False,
                "spatial_as_samples": False,
            },
            "sim_loss_t": {
                "metric_type": "temporal_similarity",
                "projector": {
                    "enabled": False,
                    "mlp_spec": None,
                    "hidden_multiplier": 4,
                },
                "after_projection": False,
            },
            "idm_loss": {
                "metric_type": "inverse_dynamics",
                "projector": {
                    "enabled": False,
                    "mlp_spec": None,
                    "hidden_multiplier": 4,
                },
                "inverse_dynamics_model": {
                    "enabled": True,
                    "hidden_dim": 256,
                    "action_dim": 2,
                },
                "after_projection": False,
            },
        },
    },
    "loss": {
        "loss_type": "weighted_metric",
        "name": "loss",
        "strict": True,
        "return_loss_terms": True,
        "metric_weights": {
            "prediction_loss": 1.0,
            "cov_loss": 8.0,
            "std_loss": 16.0,
            "sim_loss_t": 12.0,
            "idm_loss": 1.0,
        },
    },
}
