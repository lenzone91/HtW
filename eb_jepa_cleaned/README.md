# JEPA Framework

This repository is a modular framework for **JEPA experiments**, built around a
clean re-implementation of EB-JEPA's `ac_video_jepa` example.

The goal is to re-home a working action-conditioned video JEPA into a clearer
architecture so that future JEPA experiments can reuse the same generic
machinery, paradigm layer, and configuration system rather than forking a
monolithic example.

The architecture itself is ported from a prior framework migration (originally
shaped for Target Speech Extraction); its generic, domain-agnostic parts —
strict factories, Hydra-for-config-only, registries, the pillar split, and the
testing layout — are reused here, while all audio/TSE/SDE-specific content is
out of scope and removed.

Packaging, dependency declaration, environment management, and global
GitHub/project conventions are handled outside the scope of this folder.

## Main Objectives

The migration has three main objectives:

1. Introduce a clearer hierarchical source layout.
2. Move configuration composition to Hydra, without using Hydra for object
   instantiation.
3. Remove permissive factory/config behavior: all factory and config errors are
   strict by default.

The migration should expose and fix architectural/interface bugs, not hide them
through local patches.

## Target Source Layout

    eb_jepa_cleaned/            (project root: pyproject.toml, tests/)
      eb_jepa_cleaned/          (the importable package)
        Workflow/
        AIML/
        AcVideoJEPA/

## Folder Roles

### Workflow

`Workflow` owns generic Python-level project protocols.

Typical responsibilities:

- factory protocols;
- registries;
- config loading and composition helpers;
- runtime context conventions;
- generic validation utilities.

It must not depend on AIML or AcVideoJEPA.

### AIML

`AIML` owns generic AI/ML pipeline logic.

Typical responsibilities:

- datasets;
- interfaces;
- augmentations;
- collators;
- dataloaders;
- models;
- Lightning module helpers;
- metrics;
- metric sets;
- losses;
- optimizers;
- schedulers;
- trainers;
- callbacks;
- loggers;
- execution flows.

It may depend on Workflow.

It must remain domain-agnostic: it knows nothing about the JEPA objective,
video, or actions.

### AcVideoJEPA

`AcVideoJEPA` owns the concrete action-conditioned video JEPA experiment — the
re-implementation of EB-JEPA's `ac_video_jepa` example.

There is **no separate JEPA pillar**: JEPA is just an encoder plus a loss and
regularizers over embeddings (a latent-space prediction objective with
anti-collapse terms). Those are an encoder model and a set of metrics, built on
AIML's generic machinery — not a distinct architectural layer. The JEPA
objective therefore lives here, with the experiment that uses it.

Typical responsibilities:

- the JEPA objective: the latent prediction loss and the variance/covariance
  (and temporal / inverse-dynamics) regularizers, as metrics over embeddings;
- video / frame-stack datasets (e.g. two-rooms navigation);
- action conditioning (action encoders, action-conditioned predictor wiring);
- the concrete encoder backbones (e.g. Impala / RNN);
- latent rollouts (autoregressive / parallel multi-step prediction);
- planning and evaluation trajectories over the learned latent dynamics;
- the JEPA Lightning module and the experiment's Hydra config trees.

It may compose objects from Workflow and AIML, and registers its concretes onto
AIML registries at import time.

## Dependency Direction

Allowed dependency direction:

    Workflow     -> no project-specific dependency
    AIML         -> may depend on Workflow
    AcVideoJEPA  -> may depend on Workflow + AIML

Forbidden directions unless explicitly justified:

    Workflow     -> AIML / AcVideoJEPA
    AIML         -> AcVideoJEPA

The purpose is to avoid circular architecture and hidden coupling.

## Configuration Policy

Hydra is used for configuration composition only.

Hydra may handle:

- config groups;
- defaults lists;
- command-line overrides;
- interpolation;
- final config composition.

Hydra must not instantiate project objects.

The project boundary is:

    Hydra config composition
      -> resolved config
      -> plain Python dict
      -> project factories
      -> built objects

Project code should not depend on `DictConfig` or `OmegaConf` objects outside the
config handling layer.

## Factory Policy

Factories are responsible for object construction.

All factory behavior is strict:

- unknown object name -> error;
- unknown config key -> error;
- missing required key -> error;
- invalid runtime context -> error;
- wrong constructed object type -> error.

There is no `strict=False` mode.

Compatibility with legacy configs should be handled through explicit migration
logic, not silent permissive behavior.

## Runtime Context Policy

Runtime-dependent values belong in `runtime_context`.

Examples:

- device;
- run directory;
- checkpoint directory;
- resolved paths;
- logger handles;
- external runtime state.

Datasets, models, metrics, and modules should not resolve global paths or infer
runtime state internally.

## Lightning Policy

Lightning modules should receive already-built objects.

They may manage training, validation, testing, logging, and orchestration, but
they should not be responsible for low-level factory dispatch or config
resolution.

## Testing Policy

Each local logic should be unit tested.

Each flow should be integration tested.

Recommended test layout:

    tests/
      unit/
      integration/
      smoke/

Meanings:

- unit tests: one object, one local responsibility;
- integration tests: one subsystem flow;
- smoke tests: minimal end-to-end run.

A migration step is not complete just because imports work. It is complete when
the relevant tests and documentation have been updated.

## Documentation Policy

Every important folder should have a `README.md`.

Documentation should not be redundant.

High-level README files explain:

- why the folder exists;
- what it owns;
- what it must not own;
- allowed dependency directions.

Low-level README files explain:

- exact contracts;
- factory ownership;
- extension rules;
- expected inputs and outputs;
- testing expectations.

## Migration Philosophy

The migration should be incremental.

For each folder:

    move or create structure
    -> define ownership
    -> migrate or implement logic
    -> update imports
    -> write README
    -> add unit tests
    -> add integration tests when relevant
    -> update chatbot_context.md

Avoid broad rewrites unless explicitly needed.

Fix architectural bugs at their responsible subsystem, not through local patches
designed only to make a toy run pass.
