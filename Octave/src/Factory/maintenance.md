# Factory Maintenance

This file explains how to cleanly maintain the `Factory/` folder.


## 1. File roles

The `Factory/` folder centralizes reusable factory abstractions shared across the project.

```text
Factory/
- base.py
- maintenance.md
```

### base.py

Shared factory abstractions.

Responsibilities:

- define the common builder interface;
- centralize common config validation logic;
- centralize strict / non-strict error handling;
- centralize reusable factory helpers;
- propagate runtime context through builders;
- define reusable dispatcher logic for collections of config-based objects.

Currently contains:

- BaseBuilder
- BaseBuilderDispatcher

#### BaseBuilder

Shared parent class for config-based builders.

Responsibilities:

- define the common __call__ interface;
- validate config structure;
- optionally validate config keys against a default config;
- centralize error handling;
- propagate runtime_context;
- delegate object-specific construction to:
- build_from_config.

Standard builder call:

builder(
    config=config,
    runtime_context=runtime_context,
)

Expected builder method signature:

build_from_config(
    config: dict,
    runtime_context: dict | None = None,
)

runtime_context is optional.

Builders that do not require runtime information may ignore it.

This class should only contain reusable factory-level logic.

Object-specific validation or construction logic should not be added here.

#### BaseBuilderDispatcher

Shared parent class for dispatching collections of named configs to object-specific builders.

Expected pattern:

configs[object_name] = object_config

Responsibilities:

- validate dispatcher-level config structure;
- validate builder registries;
- iterate over named configs;
- dispatch each config to the associated builder;
- propagate runtime_context to child builders;
- centralize reusable dispatcher helpers.

The builder registry follows the convention:

builder_registry[object_name] = builder

The existence of a builder implies that the corresponding object is implemented and buildable.

Dispatcher-specific naming or project-specific dispatch rules should not be added here.

### maintenance.md

Factory maintenance protocol.

Responsibilities:

- document reusable factory abstractions;
- explain where new factory logic should be added;
- explain how to extend builders and dispatchers cleanly;
- document the project-wide factory interface conventions.
## 2. Runtime context propagation

Factories now support propagation of runtime information through:

runtime_context

Expected flow:

Setup/
-> runtime_context

Execution/
-> subsystem builders

Factory/
-> propagate runtime_context

Subsystem builder
-> optionally use runtime_context

Typical runtime information includes:

- resolved paths;
- run directories;
- hardware information;
- environment information;
- logger runtime state.

Factories should not assume that runtime_context is always present.

Expected signature: runtime_context: dict | None = None

This allows older builders to remain compatible while the project is progressively updated.

## 3. Adding a new builder

A new builder should inherit from:

BaseBuilder
### Step 1 : Define a default config

Each builder should rely on a default config.

Example: DEFAULT_MY_OBJECT_CONFIG

The default config defines:

- allowed keys;
- expected config structure;
- default values.

### Step 2 : Create the builder

Example:

class MyObjectBuilder(BaseBuilder):

The builder should:

- validate config-specific logic;
- implement:
- build_from_config.

Expected method signature:

def build_from_config(
    self,
    config: dict,
    runtime_context: dict | None = None,
):

### Step 3 : Use runtime_context only if needed

Builders should only use runtime_context when runtime information is actually required.

Examples:

- resolving runtime paths;
- loading checkpoints;
- accessing experiment directories.

Builders that do not require runtime information should simply ignore it.

### Step 4 : Keep builders construction-focused

Builders should:

- construct objects;
- validate configs;
- optionally inject runtime information.

Builders should not:

- execute training;
- execute evaluation;
- implement business logic;
- store global mutable state.

## 4. Adding a new dispatcher

A new dispatcher should inherit from:

BaseBuilderDispatcher

## Step 1 : Define the registry

Expected convention:

builder_registry[object_name] = builder

The registry defines which objects are buildable.

## Step 2 : Define the dispatcher config structure

Expected pattern:

configs[object_name] = object_config

## Step 3 : Reuse dispatcher propagation

Dispatchers should reuse the base dispatcher propagation logic:

builder(
    config=object_config,
    runtime_context=runtime_context,
)

Do not manually duplicate runtime propagation logic.

## Step 4 : Keep dispatchers generic

Dispatchers should only:

- validate dispatch structure;
- select builders;
- propagate configs;
- propagate runtime context.

They should not implement subsystem-specific object construction logic.

## 5. Strict vs non-strict behavior

Builders support:

strict=True
strict=False
Strict mode

Strict mode raises errors immediately.

Use strict mode for:

development;
debugging;
production pipelines.
Non-strict mode

Non-strict mode converts errors into warnings.

Use non-strict mode only intentionally.

## 6. Config validation philosophy

Default validation rule:

- user keys ⊆ default config keys

Factories validate inclusion, not exact equality.

This allows:

- partial configs;
- config inheritance;
- lightweight overrides.

Unknown keys should raise or warn depending on strict mode.



## 7. Sanity check of the factory system

After modifying Factory/, check that:

builders still accept:
- config;
- runtime_context;
- dispatchers still propagate runtime_context;
- builders still work when runtime_context=None;
- strict mode still raises errors;
- non-strict mode still warns;
- default-config key validation still works;
- dispatchers still reject unknown object names;
- builders remain construction-focused;
- execution logic has not leaked into factories.

## Core rule:

If a piece of logic is reusable across several subsystem builders, it probably belongs in Factory/.

If it is specific to one subsystem object, it probably belongs in that subsystem instead.