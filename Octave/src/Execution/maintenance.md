# Execution Maintenance

This file explains how to cleanly maintain the `Execution/` folder.


## 1. File roles

The `Execution/` folder is organized by responsibility.

```text
Execution/
- train.py
- evaluate.py
- factory.py
- reports.py
- cleanup.py
- configs.py
- maintenance.md
```

### train.py

Training entry point.

Responsibilities:

- run setup_runtime;
- initialize the training execution report;
- build training objects through Execution/factory.py;
- launch trainer.fit;
- optionally resume a Lightning training run through:
- trainer.fit(..., ckpt_path=...);
- mark the report as finished or failed;
- save the report;
- close external services.

This file should not build models, datamodules, loggers, callbacks, or trainers directly.

This file should not restore weights manually.

### evaluate.py

Evaluation entry point.

Responsibilities:

- run setup_runtime;
- initialize the evaluation execution report;
- build evaluation objects through Execution/factory.py;
- launch either:
- trainer.validate;
- trainer.test;
- store Lightning evaluation outputs in the report;
- mark the report as finished or failed;
- save the report;
- close external services.

Evaluation should use module loading, not Trainer resume.

### factory.py

Execution-level object assembly.

Responsibilities:

- build the datamodule through DataModules/factory.py;
- build the Lightning module through Modules/factory.py;
- optionally apply module loading through Loading/factory.py;
- build loggers through Loggers/factory.py;
- build callbacks through callback factories when needed;
- instantiate the Lightning Trainer;
- return execution-ready object bundles.

Current public builders:

- build_training_objects;
- build_evaluation_objects.

This file should delegate subsystem construction to subsystem factories.

### reports.py

Shared execution report utilities.

Responsibilities:

- initialize JSON-serializable execution reports;
- mark reports as finished;
- mark reports as failed;
- save reports in the run directory;
- convert common Python objects into JSON-compatible values.

Reports should not store large or non-serializable Python objects.

### cleanup.py

Execution cleanup utilities.

Responsibilities:

- close external services after execution;
- remain logger-agnostic at the public API level;
- defensively close WandB only if it was imported and initialized.

This file should not force any optional logger dependency.

### configs.py

Default execution configs.

Responsibilities:

- assemble subsystem default configs into global execution configs;
- define execution-specific Trainer defaults;
- define default loading configs;
- define default Trainer resume configs.

Current default configs:

- DEFAULT_TRAINING_EXECUTION_CONFIG;
- DEFAULT_EVALUATION_EXECUTION_CONFIG;
- DEFAULT_TRAINER_CONFIG;
- DEFAULT_EVALUATION_TRAINER_CONFIG;
- DEFAULT_RESUME_CONFIG.

Subsystem defaults should stay in their own folders.

### maintenance.md

Execution maintenance protocol.

Responsibilities:

- explain file roles;
- explain how to add execution modes;
- explain how to delete execution modes;
- document minimal consistency checks after changes.

## 2. Restoration philosophy

Execution supports two distinct restoration mechanisms.

### Module loading

Module loading restores weights into an already-built LightningModule.

Expected flow:

Modules/factory.py
-> build LightningModule

Loading/factory.py
-> restore weights into LightningModule

This mechanism is appropriate for:

- evaluation;
- inference;
- fine-tuning;
- transfer learning;
- weight initialization.
### Trainer resume

Trainer resume restores the full Lightning training state.

Expected flow:

trainer.fit(..., ckpt_path=...)

This restores:

- model weights;
- optimizer states;
- scheduler states;
- loop state;
- current epoch / global step;
- AMP state.

This mechanism is appropriate for:

- continuing interrupted training;
- extending training duration.
### Important rule

Training should not use both:

loading.module.enabled = True
resume.enabled = True

at the same time.

Reason:

- module loading restores only weights;
- Trainer resume restores the full training state.

Using both simultaneously creates ambiguous restoration behavior.

### Evaluation rule

Evaluation should use:

loading.module

and must not use:

resume
## 3. Adding a new execution mode

Adding an execution mode means adding a new entry point around a Lightning method or a custom execution procedure.

Typical examples:

- prediction;
- standalone validation;
- standalone testing;
- checkpoint evaluation;
- inference export.
### Step 1 : Create the entry-point file

Add a new file in Execution/.

Example: predict.py

The file should follow the same structure as train.py and evaluate.py.

It should:

- call setup_runtime;
- initialize an execution report;
- build execution objects through Execution/factory.py;
- launch the requested Lightning method;
- mark the report as finished or failed;
- save the report;
- close external services.
### Step 2 : Add an execution factory builder

Add a public builder in factory.py.

Example:

build_prediction_objects

The builder should return only the objects needed by the entry point.

Typical format:

{
    "trainer": trainer,
    "module": module,
    "datamodule": datamodule,
}
### Step 3 : Reuse existing utilities

Reuse:

- reports.py for reports;
- cleanup.py for external service cleanup;
- existing subsystem factories for object construction.

Do not duplicate report or cleanup logic.

### Step 4 : Add a default config if needed

Add a default config in configs.py.

Example: DEFAULT_PREDICTION_EXECUTION_CONFIG

The config should assemble existing subsystem defaults when possible.

Only execution-specific defaults should be defined in Execution/configs.py.

### Step 5 : Check naming consistency

Use consistent names across:

- entry-point function;
- factory builder;
- default config;
- report filename.

Example:

run_prediction
build_prediction_objects
DEFAULT_PREDICTION_EXECUTION_CONFIG
prediction_execution_report.json
## 4. Deleting an execution mode

Deleting an execution mode means removing its entry point and every place where it is exposed.

### Step 1 : Remove the entry-point file

Delete the corresponding file.

Example: predict.py
### Step 2 : Remove its factory builder

Remove the corresponding builder from factory.py.

Example:

build_prediction_objects

Also remove any helper that was only used by this builder.

### Step 3 : Remove its default config

If the mode has a default config in configs.py, remove it.

Example: DEFAULT_PREDICTION_EXECUTION_CONFIG
### Step 4 : Remove unused imports

Check and clean imports in:

- factory.py;
- configs.py;
- __init__.py if used;
- notebooks;
- experiment scripts.
### Step 5 : Check report filenames

If scripts or notebooks expect a report file from the deleted mode, remove or update those references.

## 5. Modifying execution reports

Reports are shared across execution modes.

Add a report field

Add a field in reports.py only if it is useful across several execution modes.

Good report fields:

- execution_type;
- status;
- run_dir;
- runtime_context;
- outputs;
- error.

Avoid storing:

- Trainer;
- LightningModule;
- DataModule;
- datasets;
- loggers;
- callbacks;
- raw tensors;
- large arrays.
- Save policy

Reports should be saved in the run directory.

Failed executions should still save a report.

Errors should be recorded with repr(error) and then re-raised by the entry-point file.

## 6. Adding an external cleanup

Cleanup should stay defensive and logger-agnostic.

### Step 1 : Add a private cleanup helper

Add a small helper in cleanup.py.

Example: close_mlflow_if_active
### Step 2 : Register it in the public cleanup function

Call it inside: close_external_services
### Step 3 : Keep optional dependencies optional

The cleanup helper should not import an optional package unless it is already active or safely available.

It should not fail when the service was not used.

## 7. Sanity check of the execution pipeline

After modifying Execution/, check that:

- train.py still calls setup_runtime;
- evaluate.py still calls setup_runtime;
- object construction still goes through Execution/factory.py;
- subsystem construction still goes through subsystem factories;
- reports are saved in the run directory;
- failed executions still save a report;
- external services are closed in finally;
- training uses trainer.fit;
- validation uses trainer.validate;
- testing uses trainer.test;
- module loading stays delegated to Loading/;
- Trainer resume still uses: trainer.fit(..., ckpt_path=...);
- training does not enable both:
    - loading.module;
    - resume;
- evaluation does not use resume;
- configs only assemble subsystem defaults and execution-specific defaults.

## Core rule:

If a change requires knowing how a model, dataset, metric, logger, callback, optimizer, or preprocessing object is internally built, it probably does not belong in Execution/.