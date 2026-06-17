# Workflow / Factory

Strict construction of objects from plain config dicts. This is the only
sanctioned object-construction mechanism in the project.

## Files

- `registry.py` — `Registry` and the declarative metadata containers
  (`RegistryEntry`, `SubBuildDeclaration`, `FieldResolution`).
- `builder.py` — `RegistryBuilder`, which validates configs and constructs
  objects declared in a registry.
- `errors.py` — `FactoryError` base, with `RegistryError` and `BuilderError`.

## Model

A **registry** is a declarative catalog for one object family ("dataset",
"model", ...). It stores, per registered name: the class, a `default_config`,
an optional routing `type_field`, declared `sub_builds`, and declared
`field_resolutions`. It builds nothing.

A **builder** wraps one registry and turns a config dict into an object:

1. deep-copy the config (inputs are never mutated);
2. strip routing/type fields (they are routing metadata, not constructor args);
3. validate (`check_default_keys`: config keys must be a subset of
   `default_config`);
4. apply `field_resolutions` (transform/derive values; resolver contract:
   `resolver(config, runtime_context, **kwargs) -> value`);
5. apply `sub_builds` (build declared dependencies via linked builders;
   methods: `one`, `named`, `phase_single_named`);
6. call `object_cls(**config, **kwargs)`.

`build_one` builds a single object (name given explicitly or via `type_field`).
`build_named` builds a dict of objects keyed by the outer config keys.

## Strictness contract

The factory is strict. There is no `strict=False` mode and no permissive
fallback. Every contract violation raises immediately:

- unknown registered name → `RegistryError`;
- duplicate registration → `RegistryError`;
- malformed registry entry → `RegistryError`;
- non-dict config, unknown config key, missing sub-build key, unknown sub-build
  method, unresolvable name → `BuilderError`.

`check_default_keys` is **not** an error escape: it is a structural declaration
of whether an entry's `default_config` is an exhaustive allow-list of accepted
keys.

## Extending

Per object family, in that family's subsystem:

1. create one module-level `Registry(object_family=...)` and one shared
   `RegistryBuilder` over it;
2. register classes with `@registry.register_class(name=..., default_config=...)`,
   declaring `type_field` / `sub_builds` / `field_resolutions` as needed;
3. expose thin factory functions that call the shared builder (do not construct
   a new builder per call).

Note: `default_config` is an allow-list and documentation of accepted keys; the
builder does not merge default values into the user config.

## Tests

- `tests/unit/Workflow/Factory/` — registry and builder behavior + the full
  strict error surface.
- `tests/integration/Workflow/Factory/` — a plain dict built into an object
  graph through linked registries (type routing, field resolution, nested
  sub-builds).
