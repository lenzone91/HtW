# eb_jepa_cleaned (package)

Source root for the JEPA Framework. This is the importable package
(`import eb_jepa_cleaned...`); the pillars below are its subpackages.

The source is split into three top-level pillars with a strict dependency
direction:

    Workflow     -> no project-specific dependency
    AIML         -> Workflow
    AcVideoJEPA  -> Workflow + AIML

Each pillar owns a distinct responsibility:

- `Workflow/`     — generic project protocols: factories, registries, config
  layer, runtime context conventions, generic validation.
- `AIML/`         — generic, domain-agnostic AI/ML pipeline logic.
- `AcVideoJEPA/`  — the concrete action-conditioned video JEPA experiment. JEPA
  is not a separate pillar: it is an encoder plus a loss and regularizers over
  embeddings, so the JEPA objective lives here with the experiment that uses it
  (video datasets, action conditioning, backbones, rollouts, the JEPA objective
  metrics, the Lightning module).

See each pillar's `README.md` for ownership and contracts. See the root
`README.md` for the full architectural rationale.

## Object definition pattern

Concrete buildable objects are defined with locality. Per subsystem folder:

    registry.py   Registry instance(s) + shared RegistryBuilder.
                  Imports only Workflow.Factory. The anchor object files import.
    base.py       Abstract/base logic only (validation, abstract methods,
                  contracts). No default config, no registration.
    <object>.py   One concrete object: the class, its *_DEFAULT_CONFIG, local
                  helpers, object-specific sub_builds/field_resolutions, and its
                  @REGISTRY.register_class(...) decorator.
    factory.py    Thin. Imports the object modules (registration side-effect)
                  and exposes build_*(...) entrypoints. No object bodies, no
                  per-object configs, no cross-object dispatch beyond build_*.

Rules:

- `concrete buildable object = object + default config + registration`
- `abstract/base object = only the shared logic it genuinely owns`
- No shared `configs.py`. Defaults live with their object.
- Thin third-party wrappers (optimizers, schedulers, callbacks) may be grouped
  per family in one file (e.g. `optimizers.py` for adam/adamw/sgd).
- Cross-object / cross-subsystem build wiring lives in `registry.py` or
  `factory.py`, not in single-object files.
- Object files contain no Hydra logic and no experiment-level composition.

Config boundary: Python `*_DEFAULT_CONFIG` are canonical object defaults and the
registry key allow-list; Hydra YAML (AcVideoJEPA level) does experiment selection,
composition, and overrides. The flow is:

    Hydra YAML -> resolved plain dict -> strict factory -> object
                  (built from its local default config + overrides)

See `migration_decision.md` (Decision 27) for rationale.
