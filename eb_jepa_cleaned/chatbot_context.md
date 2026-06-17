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

    eb_jepa_cleaned/            (project root: README, pyproject.toml, tests/)
      Configs/                  user run configs (one .yaml = one run)
      src/                      the importable package (`import src...`)
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

Phase A (standalone env/scaffold) + Phases 0-2 done. AcVideoJEPA is fully built
and runs config-driven end-to-end (Backbones, Rollout, Metrics, Module, Data;
`Workflow/Setup` runtime_context; Hydra config tree; train/resume/validate run
modes + `launch` CLI; wandb logging incl. credential-file loading). Source now
lives in `src/` (import root `src.*`); user run configs live in project-root
`Configs/` (one .yaml = one run, reusing framework groups via a `pkg://` search
path; existing-results -> `ask`). Project README is an install/usage tutorial;
the architecture doc moved to `src/README.md`. Full suite: 246 passed on
`.venv_cleaned_jepa`. Remaining is optional polish (planning/evaluation
trajectories, Execution reports).

Done:

- Phase 0: scaffold (`src/` package with pillars
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

- `Models/Modules/` — registered `AcVideoJepaModule` (subclass of
  `BaseLightningModule`): holds encoder/predictor/action-encoder in `models`,
  exposes `encode`/`encode_actions`/`predict` (+ a `predictor` property the
  rollout reads). Built through the generic `build_lightning_module` via an
  ordered field-resolution chain (`resolve_models` builds encoder → probes
  `encoder_shape` → builds predictor coupled to the encoder → identity action
  encoder; `resolve_metric_set` injects `encoder_shape` into `runtime_context`;
  `resolve_rollout`/`resolve_loss`). `configure_optimizers` builds ONE optimizer
  over `self.parameters()` (the per-model base default would miss the metric
  set's inverse-dynamics / projector params). One-step `trainer.fit` green.

- `Data/` — vendored `two_rooms` (WallDataset + env, verbatim copy, Decision 30;
  deps in the `acvideo` extra), the `TwoRoomsDataset` adapter (registered
  `two_rooms`, semantic `{states, actions, locations, wall_x, door_y, metadata}`
  samples), and `AcVideoJepaCollator` (registered `ac_video_jepa`). DataLoader
  assembly uses AIML's generic `DefaultDataModule` (no experiment datamodule).
- End-to-end smoke test (`tests/smoke/`): a full plain-dict run config built via
  `build_training_objects` runs one `trainer.fit` step on real (tiny) two-rooms
  data. GREEN. The whole architecture is validated end-to-end (suite: 211).

- `Workflow/Setup` — `build_runtime_context(setup_config)` -> `{device,
  reproducibility, paths}` (Decision 22, now done). Separate channel from config.
- `AcVideoJEPA/configs/` — Hydra config tree (config + setup/datamodule/module/
  trainer groups, `# @package <group>` headers). Config-driven run flow:
  `load_resolved_config` -> `build_runtime_context(config["setup"])` ->
  `build_training_objects(config, runtime_context)` -> `trainer.fit`. Validated
  in `tests/integration/AcVideoJEPA/`.
- Importing `src.AcVideoJEPA` registers the whole experiment.
- `AIML/Execution` run orchestration: `train.py`/`resume.py`/`validate.py`
  (`run_training`/`run_resume_training`/`run_validation`), `snapshots.py`, and
  `launch.py` — the entrypoint that composes config (Configs) -> builds
  runtime_context (Setup) -> dispatches by `--mode` or `config['run']['mode']`.
  CLI: `python -m src.AIML.Execution.launch <config_dir> --mode train
  --overwrite [key=value ...]`. Generic: registers the experiment via
  `config['run']['imports']` (dynamic import, no static AIML->experiment dep).

- wandb wired: `WandbLogger` (registered) with `save_dir` resolved to the run's
  logs dir (field resolution reading runtime_context); `Workflow/Setup.setup_wandb`
  (WANDB_MODE + optional login); Hydra `loggers` group (`none`/`csv`/`wandb`,
  default `none`; enable with `loggers=wandb`); run flows finalize wandb via
  `close_external_services` in a `finally`.
- credentials: `Workflow/Setup.setup_user_credential` loads a gitignored
  `user_credential.yaml` (`{wandb: {api_key: ...}}`) and exports
  `WANDB_API_KEY` (config: `setup.user_credential`), run before wandb so login
  reads it. Secrets never enter the runtime_context. wandb key is once-per-machine
  (cached in ~/.netrc), not per run.

Optional polish remaining: planning/evaluation trajectories (eval side of
ac_video_jepa); Execution reports.

Test entrypoint: `.venv_cleaned_jepa/Scripts/python.exe -m pytest -q` (install
`-e ".[acvideo,dev]"`).

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
- Import root: the source lives in `src/`, which IS the top-level import package
  `src` (pillars are its subpackages) — `from src.Workflow...`, `from src.AIML...`,
  `from src.AcVideoJEPA...` (`where=["."]` + `include=["src*"]`; Decision 31).
  Internal code uses relative imports. Test dirs are packages (`__init__.py`
  throughout `tests/`) to avoid duplicate-basename clashes.
- Run tests: `.venv_cleaned_jepa/Scripts/python.exe -m pytest` from the project
  root.
- Launch runs: `.venv_cleaned_jepa/Scripts/python.exe -m src.AIML.Execution.launch
  Configs --config-name <run>` (user run configs live in project-root `Configs/`).
- Full suite currently green (244+).

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
