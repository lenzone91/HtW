# Modules README

This folder owns LightningModule orchestration for AcVideoJepa.

## File Roles

`base.py`

- defines logging helpers shared by Lightning modules;
- forwards explicit Lightning logging options to `self.log_dict`;
- validates ML-step names and log dictionaries.

`ac_video_jepa_module.py`

- defines AcVideoJepa training orchestration;
- receives an already-built AcVideoJepa model;
- parses collated batch dictionaries;
- calls `model.unroll`.
- logs train losses on step and epoch;
- logs validation and test losses on epoch;
- emits scalar logs to configured Lightning loggers.

`configs.py`

- stores plain default Lightning module configs.

`factory.py`

- builds the model through `Model/ac_video_jepa/factory.py`;
- passes already-built objects into `AcVideoJepaModule`.

## Subsystem Contract

Lightning modules may:

- parse collated batch dictionaries;
- call already-built models;
- compute configured unroll objectives;
- log flat loss dictionaries;
- configure optimizers from plain configs.

Lightning modules must not:

- build model architectures directly;
- build datasets or collators;
- resolve paths;
- depend on hidden global config.

## Batch Contract

The module expects collated batches with:

```python
{
    "states": tensor,
    "actions": tensor,
    ...
}
```

`states` and `actions` are passed to AcVideoJepa `unroll`.

## Logging Contract

Training metrics are logged with:

```python
on_step=True
on_epoch=True
logger=True
```

Validation and test metrics are logged with:

```python
on_step=False
on_epoch=True
logger=True
```

This keeps progress-bar feedback and dashboard scalar curves aligned.

## Extension Steps

1. Keep model construction in the model factory.
2. Keep batch construction in collators.
3. Add module orchestration in `ac_video_jepa_module.py`.
4. Add focused module tests.
5. Update this README when training-step behavior changes.
