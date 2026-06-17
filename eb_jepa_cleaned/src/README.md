# src — JEPA Framework source

`src` is the importable package (`import src...`) for a modular **JEPA
experiments** framework, built around a clean re-implementation of EB-JEPA's
`ac_video_jepa` example. The architecture's generic, domain-agnostic parts —
strict factories, Hydra-for-config-only, registries, the pillar split, the
testing layout — are reusable beyond this one experiment.

(For installation and how to run experiments, see the project `README.md` one
level up.)

## Design objectives

1. A clear hierarchical source layout.
2. Configuration composition via Hydra, **without** using Hydra for object
   instantiation.
3. Strict factory/config behavior by default (no permissive paths).

Architectural/interface bugs are fixed at their responsible subsystem, not hidden
behind local patches.

## Source layout

    src/                  (the importable package)
      Workflow/
      AIML/
      AcVideoJEPA/

## Folder roles

### Workflow

Generic Python-level project protocols: factory protocols, registries, the Hydra
config layer, runtime-context construction (`Setup`), generic validation. Must
not depend on AIML or AcVideoJEPA.

### AIML

Generic AI/ML pipeline logic: datasets, augmentations/adaptations, collators,
dataloaders, models, Lightning-module helpers, metrics, metric sets, losses,
optimizers, schedulers, loggers, callbacks, execution flows. May depend on
Workflow. Stays domain-agnostic: it knows nothing about the JEPA objective,
video, or actions.

### AcVideoJEPA

The concrete action-conditioned video JEPA experiment (the `ac_video_jepa`
re-implementation). There is **no separate JEPA pillar**: JEPA is just an encoder
plus a loss and regularizers over embeddings, so the JEPA objective (an encoder
model + a set of metrics, built on AIML machinery) lives here with the experiment
that uses it — alongside the two-rooms data, action conditioning, the
Impala/RNN backbones, the latent rollout, the Lightning module, and the
experiment's Hydra config groups (`AcVideoJEPA/configs/`). May depend on Workflow
and AIML, and registers its concretes onto AIML registries at import time.

## Dependency direction

    Workflow     -> no project-specific dependency
    AIML         -> may depend on Workflow
    AcVideoJEPA  -> may depend on Workflow + AIML

Forbidden (unless explicitly justified): Workflow -> AIML/AcVideoJEPA;
AIML -> AcVideoJEPA. This avoids circular architecture and hidden coupling.

## Configuration policy

Hydra composes configuration only (groups, defaults lists, overrides,
interpolation) — never instantiates objects. The boundary is:

    Hydra config composition -> resolved config -> plain dict -> factories -> objects

No `DictConfig`/`OmegaConf` leaks past the config layer.

## Factory policy

Factories build objects from plain dicts, strictly: unknown object name, unknown
config key, missing required key, invalid runtime context, or wrong built type
all raise. There is no `strict=False` mode.

## Runtime-context policy

Runtime-dependent values (device, run/checkpoint directories, resolved paths,
logger handles, external state) belong in `runtime_context`, built by
`Workflow/Setup`. Datasets, models, metrics, and modules do not resolve global
paths or infer runtime state internally. Config and runtime_context are separate
channels passed to factories — never merged.

## Lightning policy

Lightning modules receive already-built objects. They orchestrate
train/val/test/logging, not factory dispatch or config resolution.

## Object definition pattern

Concrete buildable objects are defined with locality. Per subsystem folder:

    registry.py   Registry instance(s) + shared RegistryBuilder. Imports only
                  Workflow.Factory; the anchor object files import.
    base.py       Abstract/base logic only. No default config, no registration.
    <object>.py   One concrete object: the class, its *_DEFAULT_CONFIG, local
                  helpers, object-specific sub_builds/field_resolutions, and its
                  @REGISTRY.register_class(...) decorator.
    factory.py    Thin: imports the object modules (registration side-effect) and
                  exposes build_*(...) entrypoints.

- `concrete buildable object = object + default config + registration`
- `abstract/base object = only the shared logic it genuinely owns`
- No shared `configs.py`; defaults live with their object.
- Thin third-party wrappers (optimizers/schedulers/callbacks/loggers) may be
  grouped per family in one file.
- Cross-object / cross-subsystem build wiring lives in `registry.py` or
  `factory.py`, not in single-object files.
- Experiment concretes register onto AIML registries at import time
  (AcVideoJEPA -> AIML).

Config boundary: Python `*_DEFAULT_CONFIG` are canonical object defaults and the
registry key allow-list; Hydra YAML (experiment level) does selection,
composition, and overrides. The builder does **not** merge defaults — full
configs come from Hydra (or `get_default_config`). See `migration_decision.md`
(Decision 27).

## Testing & documentation

Unit-test each local logic; integration-test each subsystem flow; smoke-test a
minimal end-to-end run (`tests/{unit,integration,smoke}`). Every important folder
has a `README.md` (high-level: ownership + dependencies; low-level: contracts,
extension rules, testing expectations), without redundancy. A step is complete
only when its tests and docs are updated.
