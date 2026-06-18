# Workflow / Configs

The config layer. Composes configs with Hydra, resolves them to plain Python
dicts, and provides file conversion and reproducibility snapshots.

## Contract

    compose config        (Hydra: groups, defaults, overrides, interpolation)
    resolve interpolation
    convert to plain dict  (the boundary)
    optionally save        (snapshots)

This layer composes and resolves configs. It does **not** validate
subsystem-specific content and does **not** build objects — that is
`Workflow/Factory`.

## Files

- `compose.py` — `compose_config(config_dir, config_name, overrides)` returns a
  composed `DictConfig`; `load_resolved_config(...)` is the full entrypoint
  (compose + resolve to plain dict).
- `resolve.py` — `resolve_to_plain_dict(cfg)` is the boundary: it resolves
  interpolations and returns a plain dict. `check_plain_config(obj)` asserts no
  `DictConfig`/`ListConfig` survives.
- `conversion.py` — `load_config` / `save_config` for plain-dict YAML/JSON/TOML.
- `savings.py` — reproducibility snapshot helpers (config + runtime context).
- `run_config.py` — `resolve_run_config(path)`: resolve a *run* config given as a
  fragment folder (`Configs/<run>/`, composes its `config.yaml` entry) or a
  resolved snapshot file; composition is delegated to Hydra (`load_resolved_config`).
  `save_composed_run(...)` writes the merged `<run>.yaml` snapshot;
  `python -m src.Workflow.Configs.run_config <folder>` is the CLI.
- `errors.py` — `ConfigError`.

## Hydra policy

Hydra is used for **composition only**:

- config groups, defaults lists, command-line-style overrides, interpolation.

Hydra must **not** instantiate project objects. There is no
`hydra.utils.instantiate`; the project already has explicit factories.

## The plain-dict boundary

`resolve_to_plain_dict` is the single sanctioned way to leave the
Hydra/OmegaConf layer. After it, no `DictConfig`/`ListConfig` may reach
Factory, AIML, or AcVideoJEPA. Interpolations are resolved eagerly;
missing mandatory values (`???`) and unresolvable interpolations raise
`ConfigError`. `check_plain_config` runs as a post-condition so the boundary is
executable, not just convention.

## Relationship to the prior framework

This replaces the prior framework's custom folder-reference resolver and the
`merge`/`overrides` modules: Hydra subsumes composition, overrides, and
interpolation. `conversion.py` and `savings.py` are carried over (now raising
`ConfigError`).

## Tests

- `tests/unit/Workflow/Configs/` — conversion round-trips, interpolation
  resolution, missing-value strictness, the leakage guard.
- `tests/integration/Workflow/Configs/` — an on-disk Hydra tree (groups +
  defaults + overrides + interpolation) composed, resolved, and fed to a
  factory.
