# AIML / Models

Generic model and Lightning-module machinery.

## Subfolders

- `Models/` — the model registry and `build_model(s)`. Models are plain
  `nn.Module`s; no base class.
- `Modules/` — `BaseLightningModule` + the module registry and
  `build_lightning_module(s)`.
- `Loading/` — utilities to load weights into already-built models/modules.

`factory.py` re-exports `build_model(s)`, `build_lightning_module(s)`, and the
`load_*_if_needed` helpers.

## Composition

    model config   --build_model(s)-->        nn.Module(s)
    module config  --build_lightning_module--> LightningModule
        (a concrete module sub-builds its model/metrics/loss as built objects;
         optimizers/schedulers are kept as configs for configure_optimizers)
    built model/module + loading config --load_*_if_needed--> weights restored

## JEPA / experiment concretes

The JEPA Lightning module registers from the AcVideoJEPA pillar (Phase 4); the concrete
encoder backbones and the experiment module wiring register from AcVideoJEPA
(Phase 4). `BaseLightningModule` and the registries/loaders here are generic.

## Tests

`tests/unit/AIML/Models/...` per subfolder; `tests/integration/AIML/Models/`
wires a dummy model -> module -> optimizer/metrics/loss.
