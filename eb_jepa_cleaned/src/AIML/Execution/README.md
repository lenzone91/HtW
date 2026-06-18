# AIML / Execution

Run composition and run orchestration: turn a resolved config + runtime context
into a running train / resume / validate flow.

## Files

- `Runs/factory.py` — object **composition**: `build_training_objects` /
  `build_evaluation_objects` (datamodule + module + Lightning Trainer, with
  loggers/callbacks).
- `Runs/cleanup.py` — external-service cleanup (`close_external_services`:
  finishes an active wandb run). Each run flow calls it in a `finally` so wandb
  is finalized per run.
- `train.py` — `run_training(config, runtime_context)`: fit from scratch.
- `resume.py` — `run_resume_training(config, runtime_context, checkpoint_path)`:
  fit restoring full training state from a checkpoint.
- `validate.py` — `run_validation(config, runtime_context)`: `trainer.validate`
  (module weights loaded if `config["loading"]["module"]` is set).
- `snapshots.py` — `snapshot_execution`: writes the config + runtime_context into
  the run's `configs_dir` (no-op without a run directory).
- `launch.py` — the entrypoint: compose config (Workflow/Configs) → build
  runtime context (Workflow/Setup) → dispatch by mode. CLI + `launch(...)`.

## launch

`launch` (or `python -m src.AIML.Execution.launch <run_path> ...`) takes a **run
path** — a run folder (`Configs/<run>/`) or a resolved snapshot
(`Configs/<run>.yaml`), resolved via `Workflow/Configs.resolve_run_config`
(Hydra). It picks the mode from `--mode` or `config["run"]["mode"]` and routes to
the right flow. `--overwrite` / `--ask-overwrite` set the run-dir policy; `--ckpt`
gives the resume/validate checkpoint; trailing `key=value` args are Hydra
overrides.

It stays **generic**: the experiment's concretes are registered by importing the
modules named in `config["run"]["imports"]` (resolved dynamically), so AIML never
statically imports a concrete experiment (Decision 9).

## Dependency direction

Execution composes objects from Data, Models, Metrics, and Training, and may
depend on Workflow (Configs for composition, Setup for the runtime context). It
does not statically depend on any experiment pillar.

## Deferred

- Execution **reports** (richer run reports beyond config/runtime snapshots).
- Sweeps (W&B) (Decision 26).
