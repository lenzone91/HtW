# AIML / Execution

Run-level orchestration: compose datamodule + module + Lightning Trainer from a
resolved config.

## Subfolders

- `Runs/` — run composition (`build_training_objects` / `build_evaluation_objects`,
  trainer + callbacks) and external-service cleanup.

## Dependency direction

Execution composes objects from Data, Models, Metrics, and Training (it is the
top of the AIML stack). It receives `runtime_context` and does not construct it.

## Deferred

- The Setup-coupled run orchestration (`run_training` / `run_evaluation`,
  reports, snapshots) lands with the Setup migration (Decision 22).
- Sweeps (W&B) are deferred (Decision 26).
- The full end-to-end smoke test (the AcVideoJEPA experiment + Hydra configs) is
  Phase 4.
