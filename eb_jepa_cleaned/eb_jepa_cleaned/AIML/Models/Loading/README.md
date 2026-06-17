# AIML / Models / Loading

Generic weight-loading utilities. Restores weights into already-built models and
Lightning modules. A utility folder — no registry/factory family.

## Files

- `model_loading.py` — `load_model_state_dict` (+ `extract_state_dict`) for
  torch models.
- `module_loading.py` — `load_module_from_lightning_checkpoint` for Lightning
  modules.
- `factory.py` — config-driven dispatchers: `load_model_if_needed`,
  `load_models_if_needed`, `load_module_if_needed`.
- `configs.py` — reference loading-config templates.

## Notes

- Loading only restores weights. It does not build objects, resume Trainer
  state, or restore optimizer/scheduler states.
- Construction and loading are separate concerns: objects are built by their
  factories, then loaded here.
- `strict` in these functions is torch's `load_state_dict(strict=...)` (key
  matching), not the project's factory strictness.

## Deferred

Runtime-context-rooted relative checkpoint-path resolution (against
`project_root` / `run_dir`) is deferred to the Setup migration (Decision 22).
Paths are currently expanded/resolved against the working directory.

## Tests

`tests/unit/AIML/Models/Loading/`.
