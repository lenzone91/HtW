# Configs

User run configs. **One `.yaml` here = one run.** This is where you stack your
experiments; the framework's reusable config groups live in
`src/AcVideoJEPA/configs/` (datamodule / module / trainer / setup / loggers) and
are pulled in from here.

## Writing a run config

Copy `toy_run.yaml` and edit. A run config:

- selects the framework groups via a `defaults:` list and the `pkg://` search
  path `pkg://src.AcVideoJEPA.configs` (so you don't duplicate the groups);
- sets `setup.paths.run_name` — this is the run's identity and its results
  directory (`runs/<experiment_name>/<run_name>`);
- overrides anything else under `_self_` (e.g. `trainer.max_steps`,
  `module.ac_video_jepa.encoder.*`, `loggers` via `- loggers: wandb`).

## Launching

    python -m src.AIML.Execution.launch Configs --config-name toy_run

`--mode` (train/resume/validate) overrides `run.mode`; `--ckpt PATH` feeds
resume/validate; trailing `key=value` args are Hydra overrides, e.g.

    python -m src.AIML.Execution.launch Configs --config-name toy_run trainer.max_steps=500

## One config = one run (existing results)

`run_name` keys the results directory, so re-running the same config targets the
same results. With `existing_run_dir_policy: ask` (the convention here), a re-run
detects the existing results and **asks whether to delete them**; on yes it
removes them and runs fresh, on no it aborts. Use `--overwrite` to force deletion
without asking, or `--ask-overwrite` to force the prompt. In a non-interactive
shell, `ask` aborts rather than hang.

> Run configs are user-specific; consider keeping personal ones out of version
> control.
