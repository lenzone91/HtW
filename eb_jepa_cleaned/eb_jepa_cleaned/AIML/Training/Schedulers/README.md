# AIML / Training / Schedulers

Generic LR-scheduler machinery: thin `torch.optim.lr_scheduler` wrappers.

## Files

- `schedulers.py` — `StepLR` / `MultiStepLR` / `ExponentialLR` /
  `CosineAnnealingLR` / `ReduceLROnPlateau` + default configs + registration
  (routed by `scheduler_type`).
- `registry.py` — `SCHEDULER_REGISTRY` + `SCHEDULER_BUILDER` (anchor).
- `factory.py` — `build_schedulers(optimizer_groups, scheduler_configs)`.

## Contract

One scheduler is built per named optimizer; the optimizer is passed at build
time as the `optimizer` runtime kwarg. A scheduler config without a matching
optimizer raises (strict).

## Tests

`tests/unit/AIML/Training/Schedulers/`.
