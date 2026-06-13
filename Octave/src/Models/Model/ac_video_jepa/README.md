# AcVideoJepa Model README

This folder owns AcVideoJepa model construction.

## File Roles

`ac_video_jepa_model.py`

- defines the Octave `AcVideoJepa` wrapper around EB-JEPA `JEPA`;
- stays model-focused and free of Lightning logic.

`configs.py`

- stores plain reusable model config dictionaries.

`factory.py`

- builds model components from plain dictionaries;
- wires encoder, action encoder, predictor, regularizer, IDM, and prediction cost;
- owns architecture-dependent shape probing.

## Model Contract

The model factory builds:

```text
ImpalaEncoder
-> Identity action encoder
-> RNNPredictor
-> optional Projector
-> optional InverseDynamicsModel
-> VC_IDM_Sim_Regularizer
-> SquareLossSeq
-> AcVideoJepa
```

Only AcVideoJepa is in scope.

## Factory Contract

The factory receives a plain serializable dictionary.

It may:

- instantiate EB-JEPA architecture classes;
- run dummy shape probes needed to infer downstream component dimensions;
- merge partial configs with defaults;
- reject unknown keys early.

It must not:

- parse batches;
- construct optimizers;
- log metrics;
- read run YAML directly;
- depend on global config state.

## Extension Steps

1. Add plain defaults in `configs.py`.
2. Add construction logic in `factory.py`.
3. Add focused model factory tests.
4. Update this README when model construction changes.
