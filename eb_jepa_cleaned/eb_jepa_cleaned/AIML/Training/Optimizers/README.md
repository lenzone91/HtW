# AIML / Training / Optimizers

Generic optimizer machinery: thin `torch.optim` wrappers.

## Files

- `optimizers.py` — `SGD` / `Adam` / `AdamW` (thin subclasses) + their default
  configs + registration (routed by `optimizer_type`). Grouped per family
  (Decision 27).
- `registry.py` — `OPTIMIZER_REGISTRY` + `OPTIMIZER_BUILDER` (anchor).
- `factory.py` — `build_optimizers(parameter_groups, optimizer_configs)` and
  `build_optimizers_from_models(models, optimizer_configs)`.

## Contract

One optimizer is built per named parameter group; the parameters are passed at
build time as the `params` runtime kwarg (not part of the config). A config name
without a matching parameter group raises (strict).

## Tests

`tests/unit/AIML/Training/Optimizers/`.
