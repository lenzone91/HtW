# AcVideoJepa Model README

This folder owns AcVideoJepa architecture component construction.

## File Roles

`configs.py`

- stores plain reusable model config dictionaries.

`factory.py`

- builds architecture components from plain dictionaries;
- wires encoder, action encoder, and predictor components into callers;
- owns architecture-dependent shape probing needed by downstream factories.

## Model Contract

The model factory target is:

```text
ImpalaEncoder
-> Identity action encoder
-> RNNPredictor
```

Only AcVideoJepa architecture components are in scope. Objective components such as
projectors, inverse dynamics models, regularizers, and prediction costs belong
to `Octave/src/Metrics`.

## Factory Contract

The factory receives a plain serializable dictionary.

It may:

- instantiate EB-JEPA architecture classes;
- run dummy shape probes needed to infer downstream component dimensions;
- merge partial configs with defaults;
- reject unknown keys early.

It must not:

- expose a JEPA runtime object;
- parse batches;
- compute rollout trajectories;
- build prediction costs or regularizers;
- aggregate losses;
- construct optimizers;
- log metrics;
- read run YAML directly;
- depend on global config state.

## Extension Steps

1. Add plain defaults in `configs.py`.
2. Add construction logic in `factory.py`.
3. Add focused model factory tests.
4. Update this README when model construction changes.
