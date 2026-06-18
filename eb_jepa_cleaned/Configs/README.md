# Configs

User run configs. **One run = one folder of fragments + a composed snapshot.**

    Configs/
      toy_run/            <- editable source: YAML fragments + a config.yaml entry
        config.yaml       <- entry: a Hydra `defaults:` list composing the fragments
        setup.yaml        <- # @package setup
        datamodule.yaml   <- # @package datamodule
        module.yaml       <- # @package module
        trainer.yaml      <- # @package trainer
        loggers.yaml      <- # @package _global_  (sets `loggers`)
        run.yaml          <- # @package run       (mode / imports / checkpoint_path)
      toy_run.yaml        <- the merged/composed snapshot (generated; for repro & inspection)

The fragments are the editable source of truth; each owns one config section. The
sibling `<run>.yaml` is the fully-merged result. **Composition is done by Hydra**
(via `src/Workflow/Configs/run_config.py`) — no bespoke resolver.

## Add a run

Copy `toy_run/` to `my_run/`, set `setup.paths.run_name: my_run` in `my_run/setup.yaml`,
and edit the fragments. To (re)generate the merged snapshot:

    python -m src.Workflow.Configs.run_config Configs/my_run        # writes Configs/my_run.yaml

## Launch

Point the launcher at the run **folder** (composes it) or the **snapshot**:

    python -m src.AIML.Execution.launch Configs/toy_run             # folder
    python -m src.AIML.Execution.launch Configs/toy_run.yaml        # snapshot

- `--mode {train,resume,validate}` overrides `run.mode`.
- `--ckpt PATH` feeds resume / validate.
- `--overwrite` / `--ask-overwrite` set the existing-results policy.
- trailing `key=value` are Hydra overrides, e.g. `trainer.max_steps=500`.

## One config = one run (existing results)

`setup.paths.run_name` keys the results directory
(`runs/<experiment_name>/<run_name>/`). With `existing_run_dir_policy: ask` (the
convention here), re-running a config whose results already exist **asks whether
to delete them**; on yes it deletes and runs fresh, on no it aborts. `--overwrite`
forces deletion; in a non-interactive shell `ask` aborts rather than hang.

> Run configs are user-specific; keep personal ones out of version control if
> they contain machine-specific paths or secrets.
