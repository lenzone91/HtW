# Datasets Maintenance


This file explains how to cleanly maintain the `Datasets/` subsystem.



# 1. Folder role


`Datasets/` defines dataset-level data access logic.

Global pipeline:

```text
user config
→ Config/
→ Setup/
→ runtime_context
→ Factory/
→ built datasets
→ Datamodules/
→ Lightning execution
```

Datasets/ is responsible for:

- sample indexing;
- sample loading;
- deterministic sample preprocessing;
- dataset-specific metadata handling;
- TSE sample formatting;
- clue retrieval/pairing logic.

Datasets/ is NOT responsible for:

- DataLoader construction;
- train/val/test split logic;
- batch collation orchestration;
- stochastic augmentation policies;
- training logic;
- Lightning orchestration.

# 2. Dataset philosophy

Datasets return semantic dictionary samples.

Generic dataset convention:

{
    "input": ...,
    "target": ...,
    "metadata": {...},
}

TSE dataset convention:

{
    "mixture": ...,
    "target": ...,
    "clue": ...,        # may be None
    "metadata": {...},
}

Metadata should stay:

lightweight;
explicit;
serializable whenever possible.

# 3. File roles

Datasets/

- base.py
    Generic dataset parent classes and helpers.

- waveform.py
    Waveform-specific dataset helpers.

- clue_base.py
    Clue dataset abstractions.

- tse_base.py
    TSE dataset abstractions.

- source_separation.py
    Toy source-separation sanity-check dataset.

- factory.py
    Dataset builders and dispatchers.

- configs.py
    Default dataset configs.

- maintenance.md
    Maintenance rules for the Datasets subsystem.

# 4. Runtime context and paths

Datasets should receive already-resolved paths.

Datasets should NOT:

- resolve runtime paths;
- access Setup/ directly;
- depend on global state.

Path resolution belongs to dataset builders inside:

Datasets/factory.py

Builders may consume:

runtime_context["paths"]

Dataset configs may define either:

"path"

or:

"path_key"

where path_key is resolved through the runtime context.

# 5. Base class philosophy

Base classes define conventions and reusable helpers.

They should remain lightweight.

Avoid adding:

training logic;
augmentation orchestration;
task-specific assumptions outside their scope;
heavy hidden behavior.

Current hierarchy:

BaseDataset
- WaveformDataset
- BaseClueDataset
- BaseTSEDataset

Concrete datasets may use multiple inheritance when appropriate:

class SomeDataset(WaveformDataset, BaseTSEDataset):
    ...

# 6. Normalization philosophy

Dataset-level preprocessing should remain deterministic.

Examples:

dtype conversion;
tensor conversion;
waveform shaping;
deterministic normalization tied to raw storage.

Training-time augmentations should NOT live inside datasets.

# 7. Clue philosophy

Clue datasets are independent datasets.

They map:

lookup key -> clue sample

Typical lookup keys:

speaker_id;
sample_id;
enrollment_id.

TSE pairing logic should use composition rather than inheritance.

# 8. Adding a new dataset

Example: adding LibriMixDataset.

Steps:

Create:

librimix.py
Implement the dataset class.

Add default config inside:

configs.py

Register a builder inside:

DATASET_BUILDERS_REGISTRY
Update this maintenance file if the subsystem structure changes.

# 9. Adding a new dataset helper parent

Example: adding spectrogram helpers.

Steps:

Create:

spectrogram.py
Add only representation-specific reusable helpers.
Avoid coupling:
task semantics;
training logic;
Lightning logic.
Keep helper parents orthogonal whenever possible.