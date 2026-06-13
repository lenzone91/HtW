DEFAULT_RUNTIME_CONTEXT = {
    "paths": {
        "run_dir": "Octave/runs/ac_video_jepa",
    },
}


DEFAULT_TRAINER_CONFIG = {
    "accelerator": "cpu",
    "devices": 1,
    "fast_dev_run": True,
    "enable_checkpointing": False,
    "enable_model_summary": False,
}


DEFAULT_EXECUTION_CONFIG = {
    "runtime_context": DEFAULT_RUNTIME_CONTEXT,
    "trainer": DEFAULT_TRAINER_CONFIG,
    "loggers": {},
}
