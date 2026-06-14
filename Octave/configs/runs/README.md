# Run Configs README

This folder stores plain YAML configs for AcVideoJepa runs.

## File Roles

`ac_video_jepa_toy.yaml`

- tiny CPU smoke-run config;
- exercises DataModule and Lightning module factories;
- intended for fast integration tests, not real training quality.

`ac_video_jepa_debug.yaml`

- short CPU debug config;
- runs without `fast_dev_run`;
- exercises launch, setup, execution, CSV logging, checkpointing, and reports;
- intended for integration tests before relying on longer training runs.

`ac_video_jepa_train.yaml`

- non-toy training config for AcVideoJepa;
- enables wandb logging;
- enables last, periodic, and best-value checkpoints;
- uses the Impala-RNN architecture and Two Rooms settings from the EB-JEPA
  AcVideoJepa example.

`ac_video_jepa_validate.yaml`

- validation config for a trained AcVideoJepa checkpoint;
- loads module weights through `Loading/`;
- uses wandb logging for dashboard metrics;
- does not resume Trainer state.

## Config Contract

Run configs assemble subsystem configs:

```yaml
datamodule: ...
module:
  components_config: ...
  rollout_config: ...
  objective_config: ...
  optimizer_config: ...
  scheduler_config: ...
trainer: ...
loggers: ...
setup: ...
```

`module.components_config` contains only architecture component settings.

`module.rollout_config` contains loss-free latent rollout settings.

`module.objective_config` contains metric-set and weighted-loss settings.

`setup` owns runtime path resolution, reproducibility, and wandb registration.

Runtime-resolved paths and machine-specific facts belong in `runtime_context`,
not in these subsystem objects.

## Extension Steps

1. Add a new YAML run config.
2. Keep it plain and serializable.
3. Add or update a config-structure test for non-toy configs.
4. Add or update a smoke/integration test if the config supports a runnable path.
