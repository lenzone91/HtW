# AIML / Training

Generic, domain-agnostic training components: optimizers, schedulers, loggers,
and Lightning callbacks. All are thin wrappers around torch / Lightning classes,
grouped per family (Decision 27).

## Subfolders

- `Optimizers/` — `build_optimizers(parameter_groups, configs)` (params passed
  at build time).
- `Schedulers/` — `build_schedulers(optimizer_groups, configs)` (optimizer
  passed at build time).
- `Loggers/` — `build_loggers(configs)` (`{}` -> `False`).
- `Checkpoints/` — `build_checkpoint_callbacks(configs)` -> list.
- `EarlyStoppings/` — `build_early_stopping_callbacks(configs)` -> list.

`factory.py` re-exports all build functions.

## Deferred (Setup)

Path resolution for checkpoint `dirpath` and logger `save_dir`/`dir` against the
runtime-context paths is deferred to the Setup migration (Decision 22). Paths
are taken from config as-is for now.

## Tests

`tests/unit/AIML/Training/<family>/`.
