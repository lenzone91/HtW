# AIML / Models / Modules

Generic Lightning module machinery.

## Files

- `base.py` — `BaseLightningModule` (abstract). Holds the model dict, builds
  optimizers/schedulers from configs in `configure_optimizers`, and provides
  log-merging/prefixing helpers. No default config, no registration.
- `registry.py` — `LIGHTNING_MODULE_REGISTRY` + `LIGHTNING_MODULE_BUILDER`
  (routes by `module_type`).
- `factory.py` — thin `build_lightning_module` (exactly one) /
  `build_lightning_modules`.

## Contracts

- A concrete module subclasses `BaseLightningModule`, implements the `*_step`
  methods, and registers itself; it declares its own field resolutions to build
  its model / metrics / loss as already-built objects (Decision 15).
- Optimizers and schedulers are passed as CONFIGS, not built objects: Lightning
  builds them from the module's parameters in `configure_optimizers`.
- Construction and weight loading are separate: build the module here, then load
  weights via `Models/Loading`.

## JEPA / experiment concretes

`BaseLightningModule` is generic. The JEPA Lightning module (the
predict-in-latent-space runtime) registers from the AcVideoJEPA pillar (Phase 4); the
action-conditioned video module wiring registers from AcVideoJEPA (Phase 4;
Decision 29).

## Tests

`tests/unit/AIML/Models/Modules/` (log helpers, configure_optimizers) and
`tests/integration/AIML/Models/` (module receiving built model/metrics/loss).
