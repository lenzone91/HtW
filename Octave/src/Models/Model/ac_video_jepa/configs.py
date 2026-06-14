DEFAULT_AC_VIDEO_JEPA_COMPONENTS_CONFIG = {
    "model_type": "ac_video_jepa",
    "encoder": {
        "encoder_type": "impala",
        "width": 1,
        "stack_sizes": [16, 32, 32],
        "num_blocks": 2,
        "dropout_rate": None,
        "layer_norm": False,
        "input_channels": 2,
        "final_ln": True,
        "mlp_output_dim": 512,
        "input_shape": [2, 64, 64],
    },
    "action_encoder": {
        "action_encoder_type": "identity",
    },
    "predictor": {
        "predictor_type": "rnn",
        "hidden_size": None,
        "action_dim": 2,
        "num_layers": 1,
        "use_encoder_final_ln": True,
    },
}
