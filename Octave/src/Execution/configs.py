DEFAULT_LIGHTNING_TRAINER_CONFIG = {
    "trainer_type": "lightning",
    "max_epochs": 1,
    "accelerator": "auto",
    "devices": "auto",
    "enable_checkpointing": True,
    "enable_progress_bar": True,
    "log_every_n_steps": 1,
}


DEFAULT_EXECUTION_CONFIG = {
    "trainer": dict(DEFAULT_LIGHTNING_TRAINER_CONFIG),
}
