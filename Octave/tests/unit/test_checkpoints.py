from Octave.src.Training.Checkpoints.factory import build_checkpoint_callbacks


def test_multiple_checkpoint_callbacks_have_unique_state_keys(tmp_path):
    checkpoint_configs = {
        "last": {
            "checkpoint_type": "last",
        },
        "periodic": {
            "checkpoint_type": "periodic",
        },
        "best_val_loss": {
            "checkpoint_type": "best_value",
            "monitor": "val/loss",
            "mode": "min",
        },
    }

    runtime_context = {
        "paths": {
            "checkpoints_dir": str(tmp_path),
        }
    }

    callbacks = build_checkpoint_callbacks(
        checkpoint_configs=checkpoint_configs,
        runtime_context=runtime_context,
    )

    state_keys = [callback.state_key for callback in callbacks]

    assert len(callbacks) == 3
    assert len(state_keys) == len(set(state_keys))