# AIML / Models / Models

Generic model machinery: the model registry and build entrypoints.

## Files

- `registry.py` — `MODEL_REGISTRY` + `MODEL_BUILDER` (anchor).
- `factory.py` — thin `build_model` / `build_models`.

## Contract

A model is a plain `nn.Module` (input -> output). Situation-dependent behavior
stays out of the model. A concrete model owns its `*_DEFAULT_CONFIG` and registers
itself in its own object file. There is no model base class: models are arbitrary
`nn.Module`s.

## Domain-agnostic

Concrete models — the JEPA encoder/predictor backbones and the experiment's
concrete encoders (e.g. Impala / RNN) — are JEPA/experiment-specific and live in
the AcVideoJEPA pillar (Phase 4).

## Tests

`tests/unit/AIML/Models/Models/` — the factory (with a dummy generic model).
