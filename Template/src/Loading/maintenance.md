# Loading Maintenance

This file explains how to cleanly maintain the `Loading/` folder.


## 1. File roles

The `Loading/` folder is organized by loading target.

```text
Loading/
- model_loading.py
- module_loading.py
- factory.py
- configs.py
- maintenance.md
``` 

### model_loading.py

Torch model weight loading utilities.

Responsibilities:

- load torch checkpoints through torch.load;
- extract a model state dict from a checkpoint;
- load weights into an already-built nn.Module;

support:
- strict / non-strict loading;
- device remapping;
- explicit state dict keys.

This file should not build models.

This file should not handle Lightning checkpoints or Trainer resume logic.

### module_loading.py

Lightning module weight loading utilities.

Responsibilities:

- load Lightning checkpoints through torch.load;
- extract the LightningModule state dict;
- load weights into an already-built LightningModule;
support:
- strict / non-strict loading;
- device remapping.

This file should not build Lightning modules.

This file should not handle Trainer resume logic.

### factory.py

Config-driven loading dispatchers.

Responsibilities:

- optionally load model weights from a loading config;
- optionally load module weights from a loading config;
- dispatch to the appropriate loading utility;
- validate loading type consistency;
- resolve checkpoint paths using runtime_context["paths"];
- support relative checkpoint paths through relative_to.

Current public functions:

- load_model_if_needed;
- load_module_if_needed.

This file should not build models or Lightning modules.

### configs.py

Default loading configs.

Responsibilities:

- define default model loading configs;
- define default module loading configs;
- define relative_to for checkpoint path resolution;
- provide reusable loading config templates.

This file should not define model or module architecture defaults.

### maintenance.md

Loading maintenance protocol.

Responsibilities:

- explain file roles;
- explain how to add loading modes;
- explain how to delete loading modes;
- document minimal consistency checks after changes.

## 2. Loading philosophy

Loading/ only restores weights into already-built objects.

Expected flow:

Models/factory.py
-> build model

Loading/
-> restore weights into model

and:

Modules/factory.py
-> build LightningModule

Loading/
-> restore weights into LightningModule

Loading/ does not own architecture construction.

Loading/ does not own Trainer resume logic.

Trainer resume belongs to Execution/train.py through:

trainer.fit(..., ckpt_path=...)
## 3. Adding a new loading mode

Adding a loading mode means supporting a new checkpoint structure or object type.

Examples:

- EMA checkpoints;
- HuggingFace checkpoints;
- partial-transfer checkpoints;
- custom state-dict formats.
### Step 1 : Create a dedicated loading file if needed

If the new loading mode targets a new object category, create a dedicated file.

Example: ema_loading.py

If it only extends an existing loading format, update the relevant file instead.

### Step 2 : Implement the loading utility

The loading utility should:

- receive an already-built object;
- load the checkpoint;
- extract the appropriate state dict;
call:
- load_state_dict;
- or another explicit restoration method.

It should return the updated object.

### Step 3 : Add config-driven dispatch if needed

If the new mode should be accessible through configs, update factory.py.

Example: load_ema_model_if_needed

Keep dispatch logic thin.

### Step 4 : Add default configs

If the mode requires configuration, add a default config in configs.py.

Example: DEFAULT_EMA_LOADING_CONFIG
### Step 5 : Keep loading separate from building

Do not move architecture construction into Loading/.

Bad pattern:

Loading/
-> builds model
-> loads model

Correct pattern:

Models/
-> builds model

Loading/
-> loads weights into model
## 4. Deleting a loading mode

Deleting a loading mode means removing all places where it is exposed.

### Step 1 : Remove the loading utility

Delete the corresponding loading function or file.

### Step 2 : Remove config dispatch

Remove its dispatch entry from factory.py.

### Step 3 : Remove default configs

Remove the associated config from configs.py.

### Step 4 : Clean imports

Check and remove unused imports in:

- factory.py;
- notebooks;
- experiment scripts;
- any optional __init__.py.
### Step 5 : Remove dependencies if needed

If the deleted loading mode depended on a dedicated package, remove that dependency from the requirements files if unused elsewhere.

## 5. Strict vs non-strict loading

Strict loading:

strict=True

requires:

- identical parameter names;
- identical parameter structure.

Non-strict loading: strict=False

allows:

- partial loading;
- architecture evolution;
- transfer learning.

Default policy:

strict=True

Use non-strict loading only intentionally.

## 6. Device remapping

All loading utilities should support:

map_location

This is required because checkpoints may be:

- saved on GPU;
- loaded on CPU;
- loaded on another GPU setup.

Example:

torch.load(..., map_location="cpu")
## 7. Relation with Execution/

Loading/ restores weights.

Execution/ controls training and evaluation execution.

Expected flow:

Execution/factory.py
-> build module
-> Loading/factory.py
-> return loaded module

Trainer resume is different:

Execution/train.py
-> trainer.fit(..., ckpt_path=...)

Do not move Trainer resume into Loading/.

## 8. Sanity check of the loading pipeline

After modifying Loading/, check that:

- loading still happens on already-built objects;
- model loading only targets nn.Module;
- module loading only targets LightningModule;
- strict=True still works correctly;
- strict=False still allows partial loading;
- map_location is forwarded correctly;
- checkpoints saved on GPU can load on CPU;
- loading config dispatch still matches expected loading types;
- Trainer resume still stays outside Loading/;
- relative checkpoint paths require runtime_context;
- relative_to matches a key in runtime_context["paths"];
- absolute checkpoint paths still work without path roots.

## Core rule:

If a change is about architecture construction, it belongs in Models/ or Modules/.

If a change is about restoring weights into an existing object, it belongs in Loading/.