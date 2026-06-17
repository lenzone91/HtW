# Migration Decisions

This file records stable architectural decisions made during the JEPA-framework
migration. The architecture was ported from a prior framework migration; the
decisions below keep their original numbering so existing cross-references stay
valid. Decisions that were specific to out-of-scope domains (audio/TSE/SDE) have
been re-oriented to their JEPA equivalents or withdrawn.

Actionable tasks belong in `migration_todo.md`.
Chatbot handoff state belongs in `chatbot_context.md`.

## Decision 1 — Standalone package with a local `pyproject.toml`

Superseded by Decision 30. The original decision (no local `pyproject.toml`; packaging handled by an enclosing monorepo) was inherited from the prior framework. This project is instead a **standalone, self-contained package** with its own `pyproject.toml`.

Status: superseded by Decision 30.

## Decision 2 — Hierarchical source split

The target source hierarchy is:

    src/
      Workflow/
      AIML/
      AcVideoJEPA/

Reason: the previous flat or semi-flat organization made ownership harder to reason about. The new hierarchy separates generic Python protocols, generic ML logic, and the concrete action-conditioned video JEPA experiment. (An early plan added a separate `JEPA` pillar; it was removed — see Decision 33. The package lives in `src/`; see Decision 31.)

Status: accepted.

## Decision 3 — Workflow is the foundation

`Workflow` should be migrated before AIML, JEPA, and AcVideoJEPA.

Reason: factories, configs, registries, and runtime context conventions are used by the rest of the project.

Status: accepted.

## Decision 4 — AIML owns generic AI/ML pipeline logic

`AIML` owns generic machine-learning pipeline abstractions.

Examples:

- data abstractions;
- models;
- Lightning module helpers;
- metrics;
- losses;
- training utilities;
- execution flows.

Reason: these objects are reusable across JEPA experiments and beyond.

Status: accepted.

## Decision 5 — Superseded (no JEPA pillar)

Superseded by Decision 33. This decision made `JEPA` a separate AIML
specialization pillar; that pillar was removed. JEPA is treated as an encoder
plus a loss/regularizers over embeddings (a model and metrics on AIML
machinery), owned by the experiment that uses it.

Status: superseded by Decision 33.

## Decision 6 — Withdrawn

This decision is no longer part of the framework scope.

Status: withdrawn.

## Decision 7 — Withdrawn

This decision is no longer part of the framework scope.

Status: withdrawn.

## Decision 8 — AcVideoJEPA composes reusable layers

The concrete action-conditioned video JEPA experiment belongs under
`AcVideoJEPA`. It composes Workflow and AIML objects and owns the JEPA objective
itself (encoder + loss/regularizer metrics; Decision 33).

Reason: experiment-specific code (video datasets, action conditioning, concrete
backbones, rollouts, planning, the JEPA objective metrics) may compose the
generic layers without forcing those layers to depend on a specific experiment.
A future JEPA experiment would be a sibling pillar reusing AIML.

Status: accepted.

## Decision 9 — Strict dependency direction

Allowed dependency direction:

    Workflow     -> no project-specific dependency
    AIML         -> may depend on Workflow
    AcVideoJEPA  -> may depend on Workflow + AIML

Reason: explicit dependency direction prevents circular architecture and hidden coupling.

Status: accepted.

## Decision 10 — Hydra handles config composition only

Hydra should be used for configuration handling, not object instantiation.

Allowed Hydra responsibilities:

- config groups;
- defaults lists;
- command-line overrides;
- interpolation;
- composition.

Forbidden Hydra responsibility:

- constructing project objects through `hydra.utils.instantiate`.

Reason: the project already has explicit factories. Hydra should not bypass them.

Status: accepted.

## Decision 11 — Project code receives plain dictionaries

After Hydra composition, configs should be resolved and converted to plain Python dictionaries before entering project logic.

Reason: `DictConfig` and `OmegaConf` objects should not leak into factories, AIML objects, JEPA objects, or AcVideoJEPA constructors.

Status: accepted.

## Decision 12 — Factories build objects

Factories remain the only standard mechanism for object construction from config dictionaries.

Reason: object construction should be centralized, validated, and testable.

Status: accepted.

## Decision 13 — Factory behavior is always strict

All factory errors are strict.

Examples:

- unknown object name -> error;
- unknown config key -> error;
- missing required key -> error;
- invalid runtime context -> error;
- wrong built object type -> error.

There is no `strict=False` mode.

Reason: permissive behavior hides interface bugs and increases factory complexity.

Status: accepted.

## Decision 14 — Runtime-only values belong in runtime_context

Runtime-dependent values should be placed in `runtime_context`.

Examples:

- device;
- run directory;
- checkpoint directory;
- resolved paths;
- logger handles;
- execution-specific state.

Reason: datasets, models, metrics, and modules should not infer or resolve global runtime state internally.

Status: accepted.

## Decision 15 — Lightning modules receive built objects

Lightning modules should receive already-built datasets/models/losses/metrics/optimizers/etc. as needed.

Reason: Lightning modules should orchestrate training logic, not perform config resolution or factory dispatch.

Status: accepted.

## Decision 16 — Each logic is unit tested

Every local logic introduced or migrated should have focused unit tests.

Reason: local behavior should be validated independently from full training runs.

Status: accepted.

## Decision 17 — Each flow is integration tested

Every important subsystem flow should have an integration test.

Reason: the main source of bugs in this project is interface behavior between subsystems.

Status: accepted.

## Decision 18 — Minimal end-to-end behavior is smoke tested

A minimal example project should eventually validate the full flow:

    Hydra config
      -> plain dict
      -> factories
      -> datamodule/model/module/trainer
      -> minimal run

Reason: toy runs are useful, but automated smoke tests should catch basic regressions earlier.

Status: accepted.

## Decision 19 — Each folder has a README

Every important folder should contain a `README.md`.

Reason: folder ownership and extension contracts should be explicit.

Status: accepted.

## Decision 20 — README files should not be redundant

High-level README files should explain architectural role and ownership.

Low-level README files should explain technical contracts, extension rules, and testing expectations.

Reason: duplicated documentation becomes stale quickly.

Status: accepted.

## Decision 21 — chatbot_context.md is part of the workflow

`chatbot_context.md` should be updated regularly after meaningful migration chunks.

Reason: the migration is large and will require switching between chatbots. The file prevents wasting time and tokens rediscovering context.

Status: accepted.

## Decision 22 — Workflow/Setup (deferred in Phase 1, now done)

`Workflow/Setup` (runtime context conventions: device, paths, reproducibility, environment, credentials) was not migrated during Phase 1.

Reason: the setup/runtime-context policy was expected to change as AIML, Execution, and concrete projects took shape. Migrating it before its consumers existed would lock in a shape likely to be reworked.

Update: implemented once AcVideoJEPA needed a config-driven run. `Workflow/Setup`
provides `build_runtime_context(setup_config)` -> `{device, reproducibility,
paths}` (device resolution, RNG seeding/determinism, run-dir creation with an
existing-run-dir policy). It builds the runtime_context from the Hydra-composed
`setup` config and the live environment, kept as a separate channel from config
(Decision 14). Credentials / wandb-login and logger handles remain a documented
extension (loggers are built by AIML from config; sweeps deferred, Decision 26).

Status: accepted; Setup implemented (device/reproducibility/paths).

## Decision 23 — Strict rewrite removes all permissive plumbing

The factory and config layers are ported from the prior framework as a clean strict rewrite, not a flag flip. The `strict` constructor flag, the `strict` config key (e.g. `configs.get("strict")`), the warn-or-raise `handle_error` helper, and the `return None`/`return False` control-flow threading are all removed. Every contract violation raises a semantic error immediately (`RegistryError` / `BuilderError` / `ConfigError`, all subclasses of a base error type).

Reason: permissive paths hid interface bugs and added complexity the migration is meant to remove.

Status: accepted.

## Decision 24 — Routing/type field is stripped before key validation

In `RegistryBuilder`, the routing/type field is removed before the `default_config` key allow-list check (the prior framework did it after). An entry that uses a `type_field` for routing therefore does not need to redundantly list that key in its `default_config`.

Reason: the routing field is builder-level metadata, not a constructor argument, so it should not be subject to the constructor key allow-list.

Status: accepted.

## Decision 25 — Hydra replaces the custom config resolution layer

the prior framework's custom folder-reference resolver and the `merge`/`overrides` modules are dropped. Hydra subsumes composition, defaults, overrides, and interpolation. `conversion.py` (plain-dict file IO) and `savings.py` (snapshots) are carried over. `default_config` dictionaries remain in Python as registry allow-lists; runnable Hydra YAML config trees are authored at the AcVideoJEPA level (Phase 4), not inside Workflow/AIML.

Reason: Hydra is the robust, simpler replacement for the bespoke config machinery of the prior framework.

Status: accepted.

## Decision 26 — AIML stays domain-agnostic; tested with dummy fixtures

AIML owns generic ML machinery and only the generic concretes. All
experiment-specific concretes — the JEPA objective metrics (latent prediction
loss + anti-collapse regularizers), the encoder backbones and action
conditioning, video datasets, rollouts, and the JEPA Lightning module — belong
to the experiment pillar (AcVideoJEPA, Phase 4) and register onto AIML
registries at import time (AcVideoJEPA -> AIML; see Decision 33). Because the
concrete entries live in the experiment pillar, AIML is unit/integration tested
with small dummy generic objects registered inside the test files; the full
end-to-end smoke test is deferred to Phase 4. Sweeps (W&B) is deferred like
Setup.

Reason: keeps AIML genuinely reusable across experiments and beyond, and keeps the dependency direction clean.

Status: accepted.

## Decision 27 — Object definition pattern (locality)

Concrete buildable objects are defined with locality: one object file holds the object class, its `*_DEFAULT_CONFIG`, its local helpers, its object-specific `sub_builds`/`field_resolutions`, and its registry decorator. There is no shared `configs.py`. Abstract/base objects live in `base.py` and own only shared logic (validation, abstract methods, contracts) — no config, no registration.

Per-subsystem file roles:

    registry.py   the Registry instance(s) + shared RegistryBuilder; imports
                  only Workflow.Factory (the import anchor for object files)
    base.py       abstract/shared logic only
    <object>.py   one concrete object: class + DEFAULT_CONFIG + helpers +
                  registration
    factory.py    thin: imports object modules (registration side-effect) and
                  exposes build_* entrypoints; no object bodies, no per-object
                  configs
    README.md

Settled details:

- The Registry instance + shared builder live in a minimal `registry.py`, not in
  `factory.py`, to avoid the decorator-registry circular import and keep
  `factory.py` thin.
- Thin third-party wrappers (torch optimizers/schedulers, Lightning callbacks)
  are grouped per family in one file (e.g. `optimizers.py` registering
  adam/adamw/sgd). One-file-per-object is reserved for substantial objects.
- Cross-object / cross-subsystem build wiring is not "object logic"; it lives in
  `registry.py` or `factory.py`, never in a single-object file.
- Object files must not contain Hydra logic, global factory logic, cross-object
  dispatch, or experiment-level composition.
- Ordering inside a file (especially grouped files): each object's default config
  comes immediately before that object, interleaved
  (`DEFAULT_A`, `A`, `DEFAULT_B`, `B`, ...) rather than all configs first then
  all classes — config and object are read together.

Config boundary: Python `*_DEFAULT_CONFIG` define canonical object defaults
(and the registry key allow-list); Hydra YAML (Project level, Phase 4) defines
experiment selection/composition/overrides (authored at the AcVideoJEPA level).

Reason: locality — an object's behavior, defaults, helpers, and registration are
read and changed in one place, avoiding large centralized `configs.py` /
`factory.py` files mixing unrelated objects.

Status: accepted.

## Decision 28 — Transforms split into DataAugmentation and DataAdaptation

The prior framework's single `Data/Transforms/` folder is replaced by two sibling families
under `AIML/Data/`:

- `DataAugmentation/` — stochastic data perturbation (e.g. random masking or
  injecting noise);
- `DataAdaptation/` — adapts a dataset's output representation to a model's input
  representation (e.g. raw frames -> the tensor an encoder expects); the
  dataset->model "interface".

Both are `dict -> dict` batch transforms and share `BaseBatchTransform`, which
lives in its own utility folder `AIML/Data/BatchTransform/` (base only, no
registry/factory) so neither family depends on the other. Each family has its
own role base (`BaseAugmentation` / `BaseAdaptation`) over `BaseBatchTransform`
and its own registry (`AUGMENTATION_REGISTRY` / `ADAPTATION_REGISTRY`), so
configs select from the correct family and the pipeline can wire each at its own
stage. There is no `Transforms/` folder in this framework.

Open for later phases: the adaptation representation contract grows on
`BaseAdaptation` (source/target representation, metadata propagation), and the
exact pipeline stage at which adaptation is applied (decided when
collator/datamodule are assembled).

Reason: augmentation and representation-adaptation are conceptually distinct
roles even though both are transforms; separating them clarifies configuration
and pipeline placement.

Both families subclass `BaseBatchTransform`. Their interaction and ordering
during collation is user-defined: the concrete collator/config composes them
into the collator's ordered `batch_transforms`; the framework imposes no fixed
ordering between the families.

Status: accepted.

## Decision 29 — Superseded (JEPA objective owned by the experiment)

Superseded by Decision 33. This decision split a modality-agnostic JEPA paradigm
(its own pillar) from the AcVideoJEPA experiment. The JEPA pillar was removed;
the JEPA objective (prediction loss + regularizers, as metrics, plus the encoder
model) is owned by the experiment pillar that uses it. AIML still owns only the
generic metric machinery (`MetricSet` / `LoggableMetricSet` / `WeightedMetricLoss`).

Status: superseded by Decision 33.

## Decision 30 — Standalone package, no dependency on `eb_jepa`

This project is a standalone, installable package (`eb_jepa_cleaned`) with its
own `pyproject.toml`. It does **not** depend on, import, or inherit from the
`eb_jepa` package.

The primitives that the prior `ac_video_jepa` implementation borrowed from
`eb_jepa` (loss functions, encoder/predictor backbones, the two-rooms dataset
and environment, planning) are **re-implemented or vendored into this tree**:

- small, well-understood primitives (latent prediction loss, variance/covariance
  regularizers, Impala/RNN backbones, projector, inverse-dynamics model) are
  cleanly re-implemented against this project's conventions;
- bulky reference code (the `two_rooms` physics environment and dataset) is
  vendored (copied and import-adapted, owned here) rather than rewritten.

The `eb_jepa` repository remains a read-only reference for parity checking only.

Reason: `eb_jepa` is packaged in a way that causes import conflicts (shared
top-level module names such as `losses`, `architectures`, `datasets`). Owning
the primitives removes the runtime dependency and the conflict entirely, and
fits the "cleaner, more modular" goal.

Status: accepted.

## Decision 31 — `src/` is the importable package

The source lives in `src/`, which IS the top-level import package `src`; the
pillars are its subpackages:

    eb_jepa_cleaned/            <- project root (README, pyproject.toml, tests/, Configs/)
      src/                      <- the importable package (`import src...`)
        Workflow/ AIML/ AcVideoJEPA/

Discovered via `[tool.setuptools.packages.find] where = ["."]`,
`include = ["src*"]`. Import paths are `src.Workflow.…`, `src.AIML.…`,
`src.AcVideoJEPA.…`. Internal code uses relative imports.

Reason: the source uses deep relative imports (e.g. `from ....Workflow` reaching
across pillars), which require the pillars to share a common parent package.
`src` being that parent both satisfies the relative imports and keeps the
conventional project layout (`src/` for source, `Configs/`, `tests/` at root).
The generic top-level name `src` is acceptable inside the project's dedicated
virtual environment.

Status: accepted (revisitable: a namespaced `eb_jepa_cleaned.*` root would need a
`src/eb_jepa_cleaned/` layout or a `package-dir` mapping).

## Decision 32 — Dedicated virtual environment, deps split by pillar

Development uses a dedicated virtual environment, `.venv_cleaned_jepa`
(Python 3.12). Dependencies are declared in `pyproject.toml`:

- core (`[project.dependencies]`): the Workflow/AIML/JEPA framework needs only
  `torch`, `lightning`, `torchmetrics`, `hydra-core`, `omegaconf`, `numpy`;
- `acvideo` extra: the vendored AcVideoJEPA experiment deps (`pymunk`,
  `gymnasium`, `opencv-python`, `einops`, `wandb`), installed at Phase 4;
- `dev` extra: `pytest`.

Reason: the core framework stays installable without the heavy experiment/sim
dependencies, and the dependency split mirrors the pillar boundaries.

Status: accepted.

## Decision 33 — No JEPA pillar; JEPA is an encoder + metrics

There is no separate `JEPA` pillar. JEPA is, in essence, an encoder plus a loss
and regularizers over embeddings (a latent-space prediction objective with
anti-collapse terms). In this framework that is just:

- an **encoder model** (built through the AIML model registry), and
- a set of **metrics** (the latent prediction loss and the variance/covariance,
  temporal, inverse-dynamics regularizers) over embeddings, evaluated by an AIML
  `MetricSet` and weighted by an AIML `WeightedMetricLoss`.

None of this warrants its own architectural layer. Introducing a `JEPA` pillar
(with a generic `JepaOutput` step container and a `BaseJepaModule`) was too
restrictive and over-specialized: it forced a fixed predicted/target-latents
contract onto every experiment. That pillar and its tests were removed.

Consequences:

- Pillars are `Workflow` / `AIML` / `AcVideoJEPA`.
- The JEPA objective lives in the experiment pillar that uses it (AcVideoJEPA),
  built on AIML's generic metric machinery. A future, differently-shaped JEPA
  experiment is a sibling pillar, free to define its own step and objective.
- AIML remains domain-agnostic and owns only the generic metric/loss machinery.

This supersedes Decisions 5 and 29, and narrows Decision 8.

Reason: keep the architecture at the right level of generality. JEPA is a way to
define a loss, not a subsystem; modeling it as encoder + metrics avoids a
premature abstraction.

Status: accepted.