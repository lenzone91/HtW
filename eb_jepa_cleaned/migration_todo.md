# Migration TODO

This file tracks actionable migration tasks.

Durable architectural decisions belong in `migration_decision.md`.
Persistent chatbot handoff context belongs in `chatbot_context.md`.

## Phase 0 — Scaffold and Protocol

Status: done.

### Root files

- [x] Create `README.md`.
- [x] Create `chatbot_context.md`.
- [x] Create `migration_todo.md`.
- [x] Create `migration_decisions.md` (named `migration_decision.md`).

### Source skeleton

- [x] Create `src/`.
- [x] Create `src/README.md`.

### Top-level folders

- [x] Create `src/Workflow/`.
- [x] Create `src/AIML/`.
- [x] Create `src/JEPA/`.
- [x] Create `src/AcVideoJEPA/`.

### Tests skeleton

- [x] Create `tests/`.
- [x] Create `tests/unit/`.
- [x] Create `tests/integration/`.
- [x] Create `tests/smoke/`.
- [x] Create `tests/README.md`.

## Phase 1 — Workflow

Status: done (Factory + Configs). Setup deferred (see Decision 22).

Goal: migrate or implement generic project protocols first.

### Folders

- [x] Create `src/Workflow/README.md`.
- [x] Create `src/Workflow/Factory/README.md`.
- [x] Create `src/Workflow/Configs/README.md`.

### Factory

- [x] Define strict registry behavior.
- [x] Define strict builder behavior.
- [x] Remove any `strict=False` path.
- [x] Ensure unknown object names fail early.
- [x] Ensure unknown config keys fail early.
- [x] Ensure missing required config keys fail early (sub-build source keys).
- [~] Ensure wrong output object types fail early. Construction via
  `object_cls(...)` guarantees the type structurally; no extra check added.
  Revisit if an explicit `expected_type` guard is wanted.
- [x] Add unit tests for registry behavior.
- [x] Add unit tests for builder behavior.
- [x] Add integration test for config-to-object construction.

### Configs

- [x] Introduce Hydra config composition.
- [x] Ensure Hydra is not used for object instantiation.
- [x] Convert composed configs to plain dictionaries.
- [x] Prevent `DictConfig` / `OmegaConf` leakage outside config layer.
- [ ] Define runtime context construction boundary. (Deferred with Setup.)
- [x] Add unit tests for config loading/composition.
- [x] Add integration test for Hydra config -> plain dict -> factory.

### Setup (Decision 22 — now done)

- [x] Migrate runtime context conventions: `Workflow/Setup.build_runtime_context`
  -> `{device, reproducibility, paths}` (device resolution, RNG seeding /
  determinism, run-dir creation + existing-run-dir policy). Built from the
  Hydra-composed `setup` config; separate channel from config (Decision 14).
  Credentials / wandb-login + logger handles deferred (documented extension).

### Documentation

- [x] Document Workflow ownership.
- [x] Document Factory contracts.
- [x] Document Configs contracts.
- [x] Update `chatbot_context.md`.
- [x] Update `migration_decision.md` if new decisions are made.

## Phase 2 — AIML

Status: done (machinery + generic concretes). All experiment concretes (incl.
the JEPA objective) are in AcVideoJEPA (Phase 4).

Goal: migrate generic AI/ML pipeline abstractions.

Scope notes (see Decisions 23-26):

- AIML owns the *machinery* (bases, registries, builders, factories) plus the
  few generic concretes (`DefaultDataModule`, `MetricSet`/`LoggableMetricSet`,
  `WeightedMetricLoss`, torch optimizers/schedulers, Lightning callbacks,
  `BaseLightningModule`). All experiment concretes — the JEPA objective metrics
  (latent prediction loss + anti-collapse regularizers), the JEPA Lightning
  module, video datasets, action encoders, concrete backbones, rollouts — are
  Phase 4 (AcVideoJEPA), registering onto AIML registries at import time.
- Several AIML registries therefore ship empty of concrete entries; AIML is
  tested with dummy generic objects registered in the test files.
- The full end-to-end smoke test needs the experiment concretes and is deferred
  to Phase 4.
- Sweeps (W&B) is deferred like Setup.
- Dependency-ordered steps: Data(Datasets+Transforms) -> Data(Collators+
  DataModules) -> Metrics -> Training -> Models+Modules+Loading -> Execution.

### Structure

- [ ] Create `src/AIML/README.md`.
- [ ] Create `src/AIML/Data/README.md`.
- [ ] Create `src/AIML/Models/README.md`.
- [ ] Create `src/AIML/Metrics/README.md`.
- [ ] Create `src/AIML/Training/README.md`.
- [ ] Create `src/AIML/Execution/README.md`.

### Data

- [x] Define Dataset ownership.
- [x] Define Interface ownership (split: `DataAdaptation`, Decision 28).
- [x] Define Augmentation ownership (`DataAugmentation`, Decision 28).
- [x] Define Collator ownership.
- [x] Define DataLoader ownership (`DataModules` builds phase DataLoaders).
- [x] Add unit tests for local data logic.
- [x] Add integration tests for dataset -> collator (augmentation/adaptation)
  -> dataloader flow.

Notes:

- `Transforms/` replaced by `BatchTransform/` (shared base util) +
  `DataAugmentation/` + `DataAdaptation/` (Decision 28).
- Concrete datasets/collators/augmentations/adaptations are experiment-specific
  (Phase 3/4); only machinery + the generic `DefaultDataModule` are here. Tested
  with dummies.
- Dataset path/dtype field resolvers deferred (depend on Setup runtime-context
  contract).

### Models

- [x] Define model contracts (registry + `build_model(s)`).
- [x] Keep models as input-output `nn.Module` objects (no model base class).
- [x] Keep situation-dependent behavior outside models.
- [x] Define model loading boundary (`Models/Loading`, separate from
  construction; runtime-context path resolution deferred to Setup).
- [x] Add unit tests for model factories/contracts.

### Modules

- [x] Define Lightning module helper contracts (`BaseLightningModule`).
- [x] Ensure Lightning modules receive already-built objects (model/metrics/loss
  via field resolutions; optimizers/schedulers stay configs for Lightning).
- [x] Keep config resolution out of Lightning modules.
- [x] Add unit tests for module helpers.
- [x] Add integration tests for model/module interaction.

Notes:

- Concrete models (encoder backbones) and the experiment Lightning module
  excluded (Phase 4 AcVideoJEPA, Decision 33). Tested with dummies.
- Source bug fixed: `is_state_dict` now requires tensor values (previously any
  string-keyed dict passed, making the unwrap branches dead).

### Metrics

- [x] Define metric wrapper contracts (`BaseMetric`, domain-agnostic).
- [x] Define metric set contracts (`MetricSet` / `LoggableMetricSet`).
- [x] Define weighted loss contract (`WeightedMetricLoss`).
- [x] Add unit tests for metrics.
- [x] Add integration tests for metric set/loss flows (metric-set sub-builds
  metrics; loss over metric values — covered in unit/factory tests).

Notes:

- Only the generic `MetricSet`/`LoggableMetricSet` are here (Decision 33). The
  JEPA objective metrics (prediction loss + regularizers) register from the
  AcVideoJEPA pillar (Phase 4).
- Layout: `AIML/Metrics/{Metrics, Loss, MetricSets}`. The `Metrics/Metrics`
  double-name is awkward; flagged for possible rename.

### Training

- [x] Define optimizer factory ownership.
- [x] Define scheduler factory ownership.
- [~] Define trainer factory ownership. The Lightning Trainer is built in
  Execution (A6), not the Training subsystem (matches the prior framework).
- [x] Define callback/logger ownership (Loggers, Checkpoints, EarlyStoppings).
- [x] Add unit tests for local training factories.
- [x] Add integration tests for training object construction (covered by the
  per-family factory tests building real torch/Lightning objects).

Notes:

- All five families are thin wrappers grouped per family (Decision 27).
- Checkpoint `dirpath` and logger `save_dir`/`dir` runtime-context path
  resolution deferred to Setup (Decision 22).
- Sweeps (W&B) deferred (Decision 26).

### Execution

- [x] Define run execution flow (composition factory: `build_training_objects`
  / `build_evaluation_objects` + Trainer + callbacks).
- [~] Define validation/test/resume flow boundaries. Eval composition done;
  resume handling lives in the deferred Setup-coupled run orchestration.
- [~] Define report writing boundary. Deferred with Setup (reports/snapshots
  need runtime-context paths).
- [x] Add integration tests for execution flow (1-step `trainer.fit` with
  generic dummies — the Phase 2 capstone).

Notes:

- A6 migrated the composition factory + cleanup. The Setup-coupled run
  orchestration (`run_training`/`run_evaluation`, reports, snapshots) and Sweeps
  are deferred (Decisions 22, 26). Full end-to-end smoke is Phase 4.

### Documentation

- [x] Update AIML README files.
- [x] Update `chatbot_context.md`.
- [x] Update `migration_decision.md` if new decisions are made.

## Phase 3 — JEPA pillar (removed)

Status: withdrawn (Decision 33).

A separate JEPA pillar (a generic `JepaOutput` step container + `BaseJepaModule`
+ objective metrics) was briefly implemented, then removed as over-specialized.
JEPA is an encoder plus a loss/regularizers over embeddings, so the JEPA
objective is owned by the experiment (AcVideoJEPA, below), built on AIML's
generic metric machinery. The pillar and its tests were deleted; the suite
returned to 173 passing.

## Phase 4 — AcVideoJEPA

Status: not started.

Goal: re-home EB-JEPA's `ac_video_jepa` as the concrete experiment and validate
the architecture end-to-end. (Reference: the prior `HtW/Octave` migration.)

### Structure

- [ ] Create the experiment under `src/AcVideoJEPA/`.
- [ ] Add the experiment README.

### Experiment objects

- [x] Re-implement the JEPA objective metrics on AIML machinery
  (`AcVideoJEPA/Metrics`): latent prediction loss (auto/parallel) +
  variance/covariance/temporal-similarity/inverse-dynamics regularizers (on the
  5D `[B,C,T,H,W]` latents), registered onto the AIML metric registry. Encoder
  shape flows via `runtime_context["encoder_shape"]` for projector/IDM builds.
- [x] Re-implement the concrete encoder backbones (`Models/Backbones`): Impala /
  RNN + projector / inverse-dynamics model (Impala/RNN registered AIML models).
- [x] Migrate latent rollouts (`Models/Rollout`): autoregressive / parallel
  multi-step prediction + the `LatentRolloutOutput` structure (local registry).
- [ ] Migrate the video / frame-stack dataset(s) (e.g. two-rooms; vendor the
  WallDataset + env, Decision 30).
- [x] Migrate action conditioning (identity action encoder + action-conditioned
  predictor wiring at module-build time, in `resolve_models`).
- [x] Define the registered `AcVideoJepaModule` (`Models/Modules`, the JEPA
  Lightning step) onto the AIML Lightning-module registry; encoder shape probed
  in `resolve_models` and injected into `runtime_context` for the metric set.
  Single optimizer over all params (covers metric IDM/projector). One-step
  `trainer.fit` green.
- [x] Vendor the two-rooms dataset/env (`Data/two_rooms`, verbatim copy,
  Decision 30) + the `TwoRoomsDataset` adapter (registered `two_rooms`) + the
  `AcVideoJepaCollator` (registered `ac_video_jepa`). DataLoader assembly uses
  AIML's generic `DefaultDataModule` (no experiment datamodule).
- [ ] Migrate planning / evaluation trajectories. (Deferred / optional.)
- [x] Un-defer Setup — implemented in `Workflow/Setup` (NOT AcVideoJEPA):
  runtime_context, paths, reproducibility (Decision 22). AcVideoJEPA only
  consumes the resulting `runtime_context`.
- [x] Run orchestration in `AIML/Execution`: `train.py` (`run_training`),
  `resume.py` (`run_resume_training`, ckpt restore), `validate.py`
  (`run_validation`), `snapshots.py`, and `launch.py` (the CLI/programmatic
  entrypoint: compose -> setup -> dispatch by `--mode` or `config['run']['mode']`;
  `--overwrite`/`--ask-overwrite`/`--ckpt` + Hydra overrides). Generic: registers
  the experiment via `config['run']['imports']` (no static AIML->experiment dep).

### Flow

- [x] Build objects through factories (`build_training_objects`).
- [x] Run a minimal Lightning flow.
- [x] Add smoke test (the end-to-end capstone): `tests/smoke/` composes a full
  plain-dict run config, builds via Execution, and runs one `trainer.fit` step
  on real (tiny) two-rooms data. Green.
- [x] Add the experiment's Hydra config tree (`AcVideoJEPA/configs/`: config +
  setup/datamodule/module/trainer/loggers groups). Config-driven run validated
  end to end (`tests/integration/AcVideoJEPA/`): Hydra compose -> resolve plain
  dict -> build_runtime_context -> build_training_objects -> trainer.fit.
- [x] wandb logging wired: registered `WandbLogger` with `save_dir` resolved to
  the run logs dir; `Workflow/Setup.setup_wandb` (mode/login); `loggers` config
  group (`none`/`csv`/`wandb`); run flows finalize wandb via
  `close_external_services`. Validated offline in `tests/integration/AcVideoJEPA/
  test_wandb_logging.py`. (W&B sweeps still deferred, Decision 26.)
- [x] wandb API key from a user credential file: `Workflow/Setup`
  `setup_user_credential` loads the gitignored `user_credential.yaml`
  (`{wandb: {api_key: ...}}`) and exports `WANDB_API_KEY` (config
  `setup.user_credential`), run before wandb login. Secrets never stored in the
  runtime_context.

### Documentation

- [x] Document how AcVideoJEPA uses Workflow + AIML (pillar/subfolder READMEs).
- [x] Update `chatbot_context.md`.

## Phase 5 — Cleanup

Status: not started.

Goal: remove obsolete or transitional code.

- [ ] Remove obsolete import paths.
- [ ] Remove compatibility aliases unless explicitly needed.
- [ ] Remove dead configs.
- [ ] Remove dead tests.
- [ ] Check README consistency.
- [ ] Check dependency direction.
- [ ] Run all tests.
- [ ] Update final `chatbot_context.md`.