# Workflow / Setup

Builds the **runtime context**: runtime-determined values the rest of the system
consumes (device, reproducibility settings, run directories). Un-defers the
Setup work postponed in Decision 22.

## Files

- `device.py` — `resolve_device("auto"|"cpu"|"cuda")`.
- `reproducibility.py` — `setup_reproducibility(config)`: seeds RNGs, sets
  determinism flags; returns the applied settings.
- `paths.py` — `setup_paths(config)`: resolves and creates the run directory and
  its subdirs per the existing-run-dir policy (`fail`/`overwrite`/`ask`).
- `user_credential.py` — `setup_user_credential(config, project_root)`: loads a
  gitignored credential file (e.g. `user_credential.yaml` with
  `{wandb: {api_key: ...}}`) and exports selected entries to environment
  variables (e.g. `WANDB_API_KEY` from `[wandb, api_key]`). Strict; secret values
  never enter the returned context.
- `wandb_setup.py` — `setup_wandb(config)`: sets `WANDB_MODE`
  (online/offline/disabled) and optionally logs in with the API key env var
  (`wandb` imported lazily). Returns a serializable summary.
- `runtime_context.py` — `build_runtime_context(setup_config)`: assembles the
  `{device, reproducibility, paths, credentials}` runtime_context (plus `wandb`
  when configured). Order: paths -> credentials (export env) -> wandb (login can
  read the just-exported key).

## The config / runtime_context boundary

Hydra (via `Workflow/Configs`) composes the static config, including a `setup`
section. `build_runtime_context` turns that section + the live environment into a
plain `runtime_context` dict. The two are **separate channels** passed to
factories — never merged into one config (Decision 11/14): the runtime_context
holds non-serializable / runtime-only state (device, created paths, later logger
handles, probed `encoder_shape`). Where a built object needs both, the relevant
factory / field-resolver combines them at build time (e.g. a checkpoint
`dirpath` = a config template + `runtime_context["paths"]["run_dir"]`).

Setup owns run-directory creation (not Hydra's `@hydra.main`/`hydra.run.dir`), so
the existing-run-dir policy stays explicit and launcher-overridable.

## Dependencies

Workflow only. Depends on third-party (torch/numpy) but not on AIML or
AcVideoJEPA.

## Deferred

`wandb` mode/login registration (`setup_wandb`) and credential-file loading
(`setup_user_credential`) are handled. The `WandbLogger` itself is built by AIML
from the `loggers` config, with its `save_dir` resolved to the run's logs
directory. wandb sweeps remain deferred (Decision 26).

## wandb auth

The API key is needed **once per machine** (cached in `~/.netrc` after the first
login), not per run. Options: run `wandb login` once; or export `WANDB_API_KEY`;
or enable `setup.user_credential` to export the key from the gitignored
`user_credential.yaml` and set `setup.wandb.login=true` for the first online run.

## Tests

- `tests/unit/Workflow/Setup/` — device resolution, reproducibility summary, run
  directory creation + policy, and the assembled runtime_context.
