# DataModule Maintenance

This file explains how to cleanly maintain the `Datamodules/` folder.


## 1. File roles

The `Datamodules/` folder is organized by responsibility.

```text
Datamodules/
- base.py
- datamodules.py
- factory.py
- configs.py
- maintenance.md
```

### base.py

Shared parent DataModule class.

Responsibilities:

- store one dataset per ML phase;
- store one collator per ML phase;
- store one DataLoader config per ML phase;
- validate phase consistency;
- build phase-specific DataLoaders.

Currently contains:

- BaseDataModule

This file should only contain reusable orchestration logic.

Dataset-specific or task-specific logic should not be added here.

### datamodules.py

Concrete DataModule implementations.

Currently contains:

- DefaultDataModule

Concrete DataModules should remain thin wrappers around BaseDataModule unless a project genuinely requires special Lightning behavior.

### factory.py

DataModule construction logic.

Responsibilities:

- build datasets from dataset configs;
- build collators from collator configs;
- assemble DataModules from already-built objects.

This file should contain all DataModule factory-related logic.

### configs.py

Default DataModule configurations.

Responsibilities:

- store reusable DataLoader configs;
- store reusable collator configs;
- define reusable DataModule templates.

Experiment-specific changes should be done outside Datamodules/, by copying and modifying these defaults.

### maintenance.md

DataModule maintenance protocol.

Responsibilities:

- explain how to add DataModules;
- explain how to delete DataModules;
- document conceptual separation rules.

## 2. Conceptual role of DataModules

The DataModule is the orchestration layer : Datasets -> DataProcessing -> DataModule -> Lightning training

The DataModule owns:

- datasets;
- collators;
- DataLoader configs;
- phase-specific DataLoader construction.

The DataModule does not:

- implement preprocessing;
- implement augmentation;
- construct objectives;
- build diffusion / flow matching targets;
- build datasets or collators internally.


## 3. Adding a new DataModule

Adding a new DataModule should only affect the orchestration layer.

### Step 1 : Add the DataModule class

Add the class in:

datamodules.py

Every DataModule should inherit from:

BaseDataModule
### Step 2 : Add custom behavior only if needed

Typical valid specializations:

- custom predict_dataloader;
- multiple validation DataLoaders;
- distributed setup logic;
- online dataset initialization.

Do not duplicate generic DataLoader logic already implemented in BaseDataModule.

### Step 3 : Register the DataModule

Import the class in: factory.py

Then add it to: DATAMODULE_BUILDERS_REGISTRY

### Step 4 : Add a default config

Add a reusable default config in: configs.py

## 4. Deleting a DataModule
### Step 1 : Remove its config

Remove the DataModule config from: configs.py

### Step 2 : Remove it from the registry

Remove the DataModule from: DATAMODULE_BUILDERS_REGISTRY

Also remove unused imports.

### Step 3 : Delete the class if unused

Delete the class from: datamodules.py

only if no other code imports or uses it.

### Step 4 : Check imports

Verify that no remaining code imports the deleted DataModule.

## 5. Conceptual separation rules



Datasets:

- return semantic sample dictionaries;
- perform deterministic canonicalization;
- do not perform training-time augmentation.

DataProcessing:

- defines collators;
- performs batching;
- performs ML-step-wise preprocessing;
- performs augmentation;
- applies padding/cropping/masking policies.


Lightning DataModules:

- own train / val / test dataset objects;
- own train / val / test collator objects;
- build DataLoaders;
- attach the correct collator to each DataLoader;
- select ML-step-specific preprocessing pipelines.

DataModules should not implement preprocessing logic directly.


LightningModules:

- receive already-collated processed batches;
- perform objective construction;
- perform diffusion / flow matching preprocessing;
- construct noise targets, timesteps, score targets, etc.