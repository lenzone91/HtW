# AIML / Execution / Runs

Run composition: assemble a full run from a resolved config.

## Files

- `factory.py` — composition layer:
  - `build_training_objects(config, runtime_context=None)` -> `{trainer, module, datamodule}`;
  - `build_evaluation_objects(...)` (same, eval trainer);
  - `build_training_trainer` / `build_evaluation_trainer` / `build_trainer`;
  - `build_training_callbacks` (checkpoints + early stoppings).
- `cleanup.py` — `close_external_services()` (defensively finishes W&B).

## Config sections

`datamodule` (required), `module` (required), `loading`, `loggers`,
`checkpoints`, `early_stoppings`, `trainer`.

## What this layer does / does not do

- It composes already-migrated AIML objects and the Lightning `Trainer`. It
  receives `runtime_context`; it does NOT construct it.
- Module construction and weight loading are separate: the module is built, then
  module-level weights are restored via `Models/Loading` if enabled.

## Deferred

The top-level run orchestration from the prior framework (`run_training` /
`run_evaluation`, execution reports, config snapshots) depends on Setup
(runtime-context construction, `paths`, snapshot dirs) and is deferred to the
Setup migration (Decision 22). Sweeps (W&B) are deferred (Decision 26). The full
end-to-end smoke test with the AcVideoJEPA experiment configs is Phase 4; this
folder is exercised now with a generic-dummy 1-step `trainer.fit`.

## Tests

`tests/integration/AIML/Execution/` — compose training objects from a dummy
config and run a 1-step `trainer.fit` (dataset -> collator -> datamodule ->
module -> trainer).
