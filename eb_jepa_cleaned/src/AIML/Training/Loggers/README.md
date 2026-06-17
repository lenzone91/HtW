# AIML / Training / Loggers

Generic logger machinery: thin Lightning logger wrappers.

## Files

- `loggers.py` — `CSVLogger` / `WandbLogger` (thin subclasses) + default configs
  + registration (keyed by name).
- `registry.py` — `LOGGER_REGISTRY` + `LOGGER_BUILDER` (anchor).
- `factory.py` — `build_loggers(logger_configs)`.

## Contract

`build_loggers` returns a list of loggers, or `False` (the Lightning sentinel to
disable logging) for an empty config.

## Deferred

Resolution of `save_dir` / `dir` against the runtime-context paths is deferred
to the Setup migration (Decision 22). Paths are currently taken from config
as-is.

## Tests

`tests/unit/AIML/Training/Loggers/`.
