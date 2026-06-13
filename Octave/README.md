# Octave README

Octave is the AcVideoJepa migration area.

Only AcVideoJepa is in scope for this migration. Other JEPA variants are not
owned here.

## Folder Roles

`src/`

- main Octave source code;
- new migration code should be added here.

`configs/`

- future run configs;
- configs must stay plain serializable dictionaries or YAML files.

`tests/`

- unit and integration tests for Octave;
- generated source code should be covered by focused tests.

`launch.py`

- parses CLI run intent;
- dispatches to training or validation execution;
- may override `setup.paths.existing_run_dir_policy`;
- does not delete or create run directories directly.

`src/Execution/`

- owns training and validation runs.

`src/Models/`

- owns AcVideoJepa model construction, Lightning modules, and validation weight loading.

`src/Training/`

- owns training support factories such as optimization and checkpoint callbacks.

`src/Loggers/`

- owns CSV and wandb logger construction.

`src/Setup/`

- owns config resolution, path resolution, reproducibility, and wandb setup.

Wandb is supported for metric logging and dashboards. Wandb sweeps are not part
of this migration.

`Workflow/` is a legacy top-level helper area. New migration code belongs under
`src/` unless a reusable legacy helper is being deliberately updated.

## Code Ownership Rules

- datasets own sample access and deterministic sample canonicalization;
- collators own batching;
- datamodules own DataLoader assembly;
- model factories own AcVideoJepa architecture construction;
- Lightning modules own training-step orchestration;
- training support factories own optimizers and checkpoint callbacks;
- runtime-dependent facts belong in `runtime_context`;
- factories build objects from plain configs.

## Testing Rule

Whenever new source code is added under `src/`, add a focused test under
`tests/`.

## Launch Examples

```bash
python Octave/launch.py Octave/configs/runs/ac_video_jepa_train.yaml --mode train
python Octave/launch.py Octave/configs/runs/ac_video_jepa_train.yaml --mode train --overwrite
python Octave/launch.py Octave/configs/runs/ac_video_jepa_train.yaml --mode train --ask-overwrite
```

`--overwrite` and `--ask-overwrite` only modify the runtime config. Setup owns
the actual run-directory policy.
