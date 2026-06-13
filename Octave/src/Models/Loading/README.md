# Loading README

This folder owns weight loading into already-built runtime objects.

## File Roles

`configs.py`

- stores plain loading config dictionaries.

`module_loading.py`

- loads LightningModule state dicts from Lightning checkpoints.

`factory.py`

- resolves checkpoint paths;
- conditionally applies module loading.

## Subsystem Contract

Loading may:

- restore weights into already-built modules;
- resolve checkpoint paths through `runtime_context`;
- support validation/evaluation loading.

Loading must not:

- build models or modules;
- resume Trainer state;
- restore optimizer or scheduler state;
- run training.

## Loading Vs Resume

Loading restores weights into an already-built module.

Resume belongs to `Execution/train.py` and uses:

```python
trainer.fit(..., ckpt_path=...)
```

Use loading for validation/evaluation. Use resume to continue interrupted
training.

## Regression Coverage

The integration tests train a tiny AcVideoJepa run, save `last.ckpt`, then
validate with `loading.module` enabled. This protects the contract between
`Execution`, `Training/Checkpoints`, and `Models/Loading`.
