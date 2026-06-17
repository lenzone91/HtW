# AIML / Training / Checkpoints

Generic checkpoint machinery: thin Lightning `ModelCheckpoint` wrappers.

## Files

- `checkpoints.py` — `LastCheckpoint` / `PeriodicCheckpoint` /
  `BestValueCheckpoint` + default configs + registration (routed by
  `checkpoint_type`).
- `registry.py` — `CHECKPOINT_REGISTRY` + `CHECKPOINT_BUILDER` (anchor,
  `check_default_keys=False`).
- `factory.py` — `build_checkpoint_callbacks(checkpoint_configs)` -> list.

## Deferred

Resolution of `dirpath` against the runtime-context paths (`run_dir` /
`checkpoints_dir`) is deferred to the Setup migration (Decision 22). `dirpath`
is taken from config as-is (`None` -> Lightning default).

## Tests

`tests/unit/AIML/Training/Checkpoints/`.
