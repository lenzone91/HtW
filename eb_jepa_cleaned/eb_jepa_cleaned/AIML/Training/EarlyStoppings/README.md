# AIML / Training / EarlyStoppings

Generic early-stopping machinery: thin Lightning `EarlyStopping` wrappers.

## Files

- `early_stoppings.py` — `BestValueStagnationEarlyStopping` / `ThresholdEarlyStopping`
  / `DivergenceEarlyStopping` / `FiniteValueEarlyStopping` + default configs +
  registration (routed by `early_stopping_type`).
- `registry.py` — `EARLY_STOPPING_REGISTRY` + `EARLY_STOPPING_BUILDER` (anchor,
  `check_default_keys=False`).
- `factory.py` — `build_early_stopping_callbacks(early_stopping_configs)` -> list.

## Tests

`tests/unit/AIML/Training/EarlyStoppings/`.
