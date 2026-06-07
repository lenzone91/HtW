# DataProcessing Maintenance

This file explains how to cleanly maintain the `DataProcessing/` folder.


## 1. File roles

The `DataProcessing/` folder is organized by responsibility.

```text
DataProcessing/
- base.py
- transforms.py
- collators.py
- factory.py
- configs.py
- maintenance.md
```

### base.py

Shared parent classes for batch transforms and collators.

Responsibilities:

- define the generic transform interface:
- batch dict -> batch dict
- define the generic collator interface:
- list[sample dict] -> batch dict
- define shared validation helpers.

Currently contains:

- BaseBatchTransform
- BaseCollator

This file should only contain reusable high-level logic.

Transform-specific or collator-specific logic should not be added here.

### transforms.py

Concrete batch transforms.

Responsibilities:

- batch-level deterministic preprocessing;
- batch-level augmentation;
- metadata-safe batch processing.

Currently contains:

- AddNoiseAtSNR

Transforms operate on already-collated batch dictionaries.

Transforms should not:

- load files;
- resolve paths;
- depend on Lightning;
- construct objectives or training targets.

### collators.py

Concrete collators.

Responsibilities:

- convert list[sample dict] into batch dict;
- define batching semantics;
- apply registered transforms in order.

Currently contains:

- TSEWaveformCollator

Collators should not:

- load data;
- perform model-specific logic;
- construct diffusion / flow matching targets.

### factory.py

DataProcessing construction logic.

Responsibilities:

- build transforms from config dictionaries;
- build ordered transform pipelines;
- build collators equipped with transforms.

This file should contain all factory-related logic.

### configs.py

Default DataProcessing configurations.

Responsibilities:

- store reusable transform configs;
- store reusable collator configs;
- define train / val / test preprocessing templates.

Experiment-specific changes should be done outside DataProcessing/, by copying and modifying these defaults.

### maintenance.md

DataProcessing maintenance protocol.

Responsibilities:

- explain how to add transforms;
- explain how to add collators;
- explain how to delete DataProcessing objects;
- document consistency checks.


# 2. Adding a new transform

Adding a transform should update only the files involved in the transform pipeline.

## Step 1 : Add the transform class

Add the transform class in: transforms.py

Every transform should inherit from: BaseBatchTransform

## Step 2 : Implement the transform

The class should define: 
- __init__
- transform


The transform method should:

- validate required batch keys;
- validate tensor structure if needed;
- apply the batch transformation;
- return the transformed batch dictionary.

Transforms may mutate the batch dictionary in-place.

## Step 3 : Add reusable checks if needed

If a validation helper is likely to be reused by several transforms or collators, add it to: base.py

Rule:

- reusable logic goes in base.py;
- transform-specific logic stays in the transform class.

## Step 4 : Register the transform in factory.py

Import the transform class in: factory.py

Then add it to: TRANSFORM_BUILDERS_REGISTRY

The transform name should be lowercase and consistent with the rest of the project.

## Step 5 : Add a default config

Add a default config dictionary in: configs.py

Then optionally add it to a default transform pipeline config.

# 3. Adding a new collator

Adding a collator should update only the batching pipeline.

## Step 1 : Add the collator class

Add the collator class in: collators.py

Every collator should inherit from: BaseCollator

## Step 2 : Implement the batching logic

The class should define: collate_samples

The method should:

- receive: list[dict]
- collate batchable values: typically tensors
- preserve semantic batch structure.

Typical output:

{
    "mixture": torch.Tensor,
    "target": torch.Tensor,
    "metadata": list[dict],
}

Padding and masking should remain explicit.

Do not silently pad or crop unless the collator is specifically designed for it.

## Step 3 : Register the collator in factory.py

Import the collator class in: factory.py

Then add it to: COLLATOR_BUILDERS_REGISTRY

## Step 4 : Add a default config

Add a reusable default config in: configs.py

# 4. Deleting a transform

## Step 1 : Remove its config

Remove the transform config from: configs.py

if it appears in default pipelines.

## Step 2 : Remove it from the registry

Remove the transform from: TRANSFORM_BUILDERS_REGISTRY

Also remove unused imports.

## Step 3 : Delete the class if unused

Delete the transform class from: transforms.py

only if no other code imports or uses it.

## Step 4 : Check imports

Verify that no remaining code imports the deleted transform.

# 5. Deleting a collator
## Step 1 : Remove its config

Remove the collator config from: configs.py

## Step 2 : Remove it from the registry

Remove the collator from: COLLATOR_BUILDERS_REGISTRY

## Step 3 : Delete the class if unused

Delete the collator class from: collators.py

only if no other code imports or uses it.

## Step 4 : Check imports

Verify that no remaining code imports the deleted collator.

# 6. Conceptual separation rules

## Datasets/

Datasets:

- return semantic sample dictionaries;
- perform deterministic canonicalization;
- do not perform training-time augmentation.

## DataProcessing/

DataProcessing:

- performs batching;
- performs ML-step-wise preprocessing;
- performs augmentation;
- applies padding/cropping/masking policies.

### DataModules/

Lightning DataModules:

- own train / val / test dataset objects;
- own train / val / test collator objects;
- build DataLoaders;
- select ML-step-specific preprocessing pipelines : attach the correct collator to each DataLoader;.

DataModules should not implement preprocessing logic directly.

## LightningModules/

LightningModules:

- perform objective construction;
- perform diffusion / flow matching preprocessing;
- construct noise targets, timesteps, score targets, etc.