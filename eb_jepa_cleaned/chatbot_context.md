# Chatbot Context

This file is the persistent handoff context for chatbot sessions working on this
project.

It should be updated regularly during the migration so that a new chatbot can
resume efficiently without rediscovering the project philosophy, current state,
and architectural decisions.

This file should stay concise and operational. Detailed subsystem contracts
belong in local `README.md` files.

## Project

The active project is a modular framework for **JEPA experiments**. Its first
concrete goal is a clean re-implementation of EB-JEPA's `ac_video_jepa` example
(action-conditioned video JEPA).

The architecture is ported from a prior framework migration (originally shaped
for Target Speech Extraction). Only its generic, domain-agnostic parts are
reused; all audio/TSE/SDE-specific content is out of scope.

The main objective is to reorganize the project architecture so future JEPA
experiments reuse shared machinery instead of forking a monolithic example.

The main folder is assumed to be managed inside a larger project/GitHub
environment. Packaging and global project management files are outside the scope
of this conversation.

## Migration Objectives

Current migration objectives:

1. Introduce a hierarchical source split.
2. Move config composition to Hydra.
3. Use Hydra only for config handling, not object instantiation.
4. Convert resolved Hydra configs to plain Python dictionaries before entering
   project logic.
5. Keep factories responsible for object construction.
6. Remove all non-strict factory/config error handling.
7. Unit test each local logic.
8. Integration test each subsystem flow.
9. Document each folder with a non-redundant `README.md`.

## Target Source Layout

    eb_jepa_cleaned/            (project root: pyproject.toml, tests/)
      eb_jepa_cleaned/          (the importable package)
        Workflow/
        AIML/
        AcVideoJEPA/

There is no separate JEPA pillar (Decision 33): JEPA is an encoder + a
loss/regularizers over embeddings, so the JEPA objective is owned by the
experiment (AcVideoJEPA), built on AIML's generic machinery.

## Folder Roles

### Workflow

Owns generic project protocols:

- factories;
- registries;
- config loading/composition helpers;
- runtime context conventions;
- generic validation logic.

It must not depend on AIML or AcVideoJEPA.

### AIML

Owns generic AI/ML pipeline logic:

- data abstractions;
- models;
- Lightning module helpers;
- metrics;
- losses;
- training utilities;
- execution flows.

It may depend on Workflow.

It must remain domain-agnostic (no video, action, or JEPA-objective specifics).

### AcVideoJEPA

Owns the concrete action-conditioned video JEPA experiment, including the JEPA
objective itself:

- the JEPA objective: latent prediction loss + variance/covariance/temporal/
  inverse-dynamics regularizers, as metrics over embeddings (on AIML machinery);
- video / frame-stack datasets (e.g. two-rooms);
- the encoder model, action encoders, action-conditioned predictor;
- concrete encoder backbones (Impala / RNN);
- latent rollouts, planning, evaluation trajectories;
- the JEPA Lightning module and the experiment's Hydra config trees.

It may depend on Workflow and AIML, and registers concretes onto AIML registries.

## Dependency Rules

Allowed:

    Workflow     -> no project-specific dependency
    AIML         -> Workflow
    AcVideoJEPA  -> Workflow + AIML

Forbidden unless explicitly justified:

    Workflow     -> AIML / AcVideoJEPA
    AIML         -> AcVideoJEPA

## Configuration Policy

Hydra is used for config composition only.

Hydra should handle:

- config groups;
- defaults;
- command-line overrides;
- interpolation;
- final config composition.

Hydra should not instantiate project objects.

Project boundary:

    Hydra composed config
      -> resolved config
      -> plain dict
      -> factories
      -> objects

No `DictConfig` or `OmegaConf` object should leak into AIML, AcVideoJEPA, or
object constructors.

## Factory Policy

Factories build objects from plain dictionaries.

All factory errors are strict.

There is no `strict=False` mode.

Errors should be semantic and early:

- unknown object name;
- unknown config key;
- missing required key;
- invalid runtime context;
- wrong built object type.

## Runtime Context Policy

Runtime-only information belongs in `runtime_context`.

Examples:

- device;
- run directory;
- checkpoint directory;
- resolved filesystem paths;
- logger handles;
- execution-specific state.

Datasets, models, metrics, and modules should not perform global path
resolution.

## Testing Policy

Use pytest-style tests.

Expected test levels:

    tests/unit/
    tests/integration/
    tests/smoke/

Unit tests should check local logic.

Integration tests should check subsystem flows.

Smoke tests should check minimal end-to-end execution.

A folder migration is not complete unless its relevant tests pass.

## Documentation Policy

Each important folder should have a `README.md`.

Top-level README files give architectural overview.

Low-level README files give technical contracts and extension rules.

Avoid duplicating the same text across README files.

## Current Migration Phase

Phase A (standalone env/scaffold) + Phases 0-2 done. AcVideoJEPA (the concrete
experiment, incl. the JEPA objective) is next. Full suite: 173 passed on
`.venv_cleaned_jepa`.

Done:

- Phase 0: scaffold (`eb_jepa_cleaned/` package with pillars
  `Workflow`/`AIML`/`AcVideoJEPA`, `tests/{unit,integration,smoke}`).
- Phase 1 Workflow:
  - `Workflow/Factory` — strict `Registry` + `RegistryBuilder`, semantic error
    family (`FactoryError`/`RegistryError`/`BuilderError`). No `strict` flag.
  - `Workflow/Configs` — Hydra composition + `resolve_to_plain_dict` boundary +
    `conversion`/`savings`.
  - `Workflow/Setup` deferred (Decision 22).
- Phase 2 AIML (machinery + generic concretes only; strict rewrite):
  - `Data/` — `Datasets`, `BatchTransform` (shared base) +
    `DataAugmentation`/`DataAdaptation` (Decision 28), `Collators`,
    `DataModules` (`DefaultDataModule`).
  - `Metrics/` — `Metrics` (`BaseMetric`), `Loss` (`WeightedMetricLoss`),
    `MetricSets` (`MetricSet`/`LoggableMetricSet`).
  - `Training/` — Optimizers, Schedulers, Loggers, Checkpoints, EarlyStoppings.
  - `Models/` — `Models` registry, `Modules` (`BaseLightningModule`), `Loading`.
  - `Execution/Runs` — composition factory (`build_training_objects` + Trainer)
    + cleanup.

Deferred (recurring): runtime-context path resolution (dataset/checkpoint/logger/
loading paths) and the Setup-coupled run orchestration
(`run_training`/`run_evaluation`, reports, snapshots) land with Setup
(Decision 22). Sweeps (W&B) deferred (Decision 26).

Removed: a short-lived `JEPA` pillar attempt (generic `JepaOutput` +
`BaseJepaModule`) was deleted as over-specialized — JEPA is just an encoder +
metrics, owned by the experiment (Decision 33).

AcVideoJEPA (in progress) — the concrete experiment, owns the JEPA objective.
Done so far (suite 201 passing):
- `Models/Backbones/` — `ImpalaEncoder`, `RNNPredictor` (registered AIML models);
  `Projector`, `InverseDynamicsModel` (plain blocks). Re-implemented, no `eb_jepa`.
- `Models/Rollout/` — `LatentRollout` (auto/parallel) + `LatentRolloutOutput`
  (the structure metrics consume), on a local `ROLLOUT_REGISTRY`.
- `Metrics/` — prediction loss (auto/parallel) + 5D regularizers (variance /
  covariance / temporal-similarity / inverse-dynamics) on AIML metric machinery;
  `RegularizerProjectionMixin` for the `[B,C,T,H,W]` reshaping; projector/IDM
  field resolvers read `runtime_context["encoder_shape"]`. Uniform metric
  signature `forward(rollout_output, actions=None)`.

Still to do:
- `Models/Modules/` — registered `AcVideoJepaModule` (subclass of
  `BaseLightningModule`): holds encoder/predictor/action-encoder in `models`,
  exposes `encode`/`encode_actions`/`predict` for the rollout, builds
  metric_set + loss via field resolutions, probes encoder shape into
  `runtime_context`, and runs the step `rollout -> metric set -> weighted loss`.
- Data — vendor `two_rooms` (WallDataset + env), collator, datamodule.
- `Workflow/Setup` — un-defer runtime_context/paths/reproducibility/launch
  (Decision 22); AcVideoJEPA only consumes the resulting runtime_context.
- Hydra config tree + end-to-end smoke test.

Important builder note: the builder does NOT merge `default_config` (it is the
allow-list); full configs come from Hydra (or `get_default_config` in tests).
Metrics with a mandatory sub-build (`prediction_cost`) or structural field
(`inverse_dynamics_model`) must have those provided.

A prior, less-modular migration of `ac_video_jepa` exists as a reference
(`HtW/Octave`); it is the source of the concrete logic to be re-homed.

## Environment / Running Tests

- Standalone package `eb_jepa_cleaned` with its own `pyproject.toml`; no
  dependency on `eb_jepa` (Decision 30).
- Dedicated venv: `.venv_cleaned_jepa` (Python 3.12). Interpreter at
  `.venv_cleaned_jepa/Scripts/python.exe`. Install: `pip install -e ".[dev]"`
  (add the `acvideo` extra at Phase 4 for the vendored two-rooms deps).
- Import root: a single top-level package `eb_jepa_cleaned` with the pillars as
  subpackages — `from eb_jepa_cleaned.Workflow...`, `from eb_jepa_cleaned.AIML...`,
  etc. (project/package layout, `where=["."]` + `include=["eb_jepa_cleaned*"]`;
  Decision 31). Internal code uses relative imports. Test dirs are packages
  (`__init__.py` throughout `tests/`) to avoid duplicate-basename clashes.
- Run tests: `.venv_cleaned_jepa/Scripts/python.exe -m pytest` from the project
  root.
- Inherited suite (~173 tests, generic Workflow/AIML) is being re-verified in
  this repo as the Phase A baseline gate.

## Conventions

- Object definition pattern (Decision 27): per buildable object, one file with
  class + `*_DEFAULT_CONFIG` + helpers + registration; `registry.py` is the
  import anchor (Registry + shared builder); `factory.py` is thin; `base.py` is
  abstract-only. Config goes immediately before its object (interleaved). Thin
  third-party wrappers grouped per family.
- AIML registries are populated by concretes that register at import time;
  factories import their object modules for the registration side effect. The
  AcVideoJEPA pillar registers its concretes (the JEPA objective metrics, the
  encoder, the Lightning module) the same way.

## Current Decisions

- Standalone package with its own `pyproject.toml`; no dependency on `eb_jepa`
  (primitives re-implemented/vendored here) — Decisions 30/31/32.
- The source split is `Workflow` / `AIML` / `AcVideoJEPA` (Decision 33: no JEPA
  pillar — JEPA is an encoder + metrics owned by the experiment).
- Workflow is migrated first, AIML second, AcVideoJEPA third.
- Factories are always strict.
- Hydra handles configuration only, not instantiation.
- The JEPA objective (latent prediction loss + regularizers, as metrics, plus
  the encoder model) lives in AcVideoJEPA, on AIML's generic metric machinery.

## Expected Chatbot Behavior

Be brief, rigorous, and implementation-aware.

Before proposing code:

- inspect existing files when available;
- ask for missing files instead of inventing unseen code;
- preserve project conventions;
- prefer small targeted changes;
- justify folder ownership;
- distinguish source bugs from test bugs.

When generating code:

- provide only the relevant file or block;
- avoid broad rewrites unless explicitly requested;
- keep readability above cleverness;
- add comments only where they clarify design decisions.

When progressing through the migration:

- update `chatbot_context.md` after meaningful chunks;
- keep `migration_todo.md` current;
- record durable decisions in `migration_decision.md`;
- ensure unit/integration tests are added when logic is introduced.
