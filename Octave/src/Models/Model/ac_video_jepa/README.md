# AcVideoJepa Model README

This folder owns AcVideoJepa architecture block construction.

## File Roles

`ac_video_jepa_model.py`

- is being migrated away from the monolithic EB-JEPA `JEPA` wrapper;
- should not own training objectives, metrics, or rollout policy.

`blocks.py`

- defines the registered architecture block container;
- exposes encoder, action encoder, predictor, and probed encoder shape;
- stays free of rollout and objective behavior.

`configs.py`

- stores plain reusable model config dictionaries.

`factory.py`

- builds architecture blocks from plain dictionaries;
- wires encoder, action encoder, and predictor blocks;
- owns architecture-dependent shape probing needed by downstream factories.
- temporarily keeps the legacy monolithic builder until rollout and objective
  orchestration have migrated.

## Model Contract

The model factory target is:

```text
ImpalaEncoder
-> Identity action encoder
-> RNNPredictor
```

Only AcVideoJepa architecture blocks are in scope. Objective components such as
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
