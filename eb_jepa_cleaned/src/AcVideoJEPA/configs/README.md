# AcVideoJEPA / configs

The runnable Hydra config tree for the experiment (Decision 25: config trees are
authored at the experiment level). Hydra is used for **composition only**; it
does not instantiate objects.

## Layout

    config.yaml                 entrypoint (defaults list)
    setup/default.yaml          -> config.setup     (Workflow/Setup)
    datamodule/two_rooms_toy.yaml -> config.datamodule (DefaultDataModule)
    module/ac_video_jepa.yaml   -> config.module    (AcVideoJepaModule)
    trainer/toy.yaml            -> config.trainer   (Lightning Trainer kwargs)

Each group file declares its package with `# @package <group>` so the composed
config has the section keys the factories expect.

## Flow

    load_resolved_config(configs/, "config", overrides)   # Workflow/Configs (Hydra)
      -> plain dict {setup, datamodule, module, trainer}
    build_runtime_context(config["setup"])                # Workflow/Setup
      -> runtime_context {device, reproducibility, paths}
    build_training_objects(config, runtime_context)       # AIML/Execution
      -> {trainer, module, datamodule}
    trainer.fit(module, datamodule)

`runtime_context` is a separate channel from the config (Decision 14): the static
config is composed by Hydra; the live runtime values are built by Setup. They are
not merged.

## Overrides

Hydra-style overrides apply at compose time, e.g.
`load_resolved_config(dir, "config", ["trainer.max_steps=100", "setup.reproducibility.seed=0"])`.
