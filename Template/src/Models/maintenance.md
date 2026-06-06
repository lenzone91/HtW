# Models Maintenance

This file explains how to maintain the `Models/` folder.

## 1. File roles

```text
Models/
- waveUnet.py
- configs.py
- factory.py
- maintenance.md
```

### waveUnet.py

Contains the WaveUNet implementation.

For now, this model is mainly used as a sanity-check model for the data module
and Lightning pipeline.

### configs.py

Stores reusable model configs.

Configs should be plain serializable dictionaries.

### factory.py

Maps model names to model classes and builds models from configs.

## 2. Adding a model
1. Add the model class in its own file.
2. Import it in factory.py.
3. Add it to DEFAULT_MODEL_CLASSES.
4. Add a default config in configs.py if useful.

## 3. Design rules
- Do not put experiment-specific configs here.
- Keep configs serializable.
- Keep model construction separate from Lightning modules.
- Lightning modules receive already-built model objects. But Lightning module factories may build these objects from config and rely on Models/factory.py and Models/configs.py.