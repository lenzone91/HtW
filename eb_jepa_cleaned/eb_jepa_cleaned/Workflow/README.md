# Workflow

Generic project protocols. The foundation pillar: every other pillar builds on
it, and it depends on nothing project-specific.

## Owns

- **Factory** — strict registries and the registry-aware builder. The single
  mechanism for turning a plain config dict into a constructed object.
- **Configs** — the Hydra config layer: composition, the resolve-to-plain-dict
  boundary, file conversion, and reproducibility snapshots.
- *(deferred)* **Setup** — runtime context conventions (device, paths,
  reproducibility, credentials). Postponed until its consumers exist; see
  `migration_decision.md`.

## Must not own

- any AI/ML or experiment-specific logic.

## Dependencies

Workflow must not depend on AIML, or AcVideoJEPA. It depends only on
third-party libraries (Hydra/OmegaConf, PyYAML).

## The two boundaries Workflow enforces

    Hydra config  --Configs.resolve_to_plain_dict-->  plain dict
    plain dict    --Factory.RegistryBuilder.build-->  built object

`Configs` produces plain dicts and never builds objects. `Factory` consumes
plain dicts and never composes configs. Nothing downstream sees `DictConfig`.

See `Factory/README.md` and `Configs/README.md` for contracts.
