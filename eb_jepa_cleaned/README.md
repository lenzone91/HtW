# eb_jepa_cleaned

A clean, modular framework for **JEPA experiments**, built around a from-scratch
re-implementation of EB-JEPA's `ac_video_jepa` example (an action-conditioned
video JEPA: learning a latent world model of an agent navigating the *two-rooms*
environment from video + actions).

It is **standalone** — it does not depend on the original `eb_jepa` package; the
primitives it needs are re-implemented and the physics environment is vendored.
The design favors reuse for future JEPA experiments: a strict factory/registry
layer, Hydra for configuration composition only, runtime-context separation, and
unit/integration/smoke tests throughout.

## Layout

    eb_jepa_cleaned/            (project root)
      README.md                this file
      Configs/                 user run configs (one .yaml = one run)
      src/                     the importable package (Workflow / AIML / AcVideoJEPA)
      tests/                   unit / integration / smoke tests
      pyproject.toml
      user_credential.yaml     (optional, gitignored) secrets, e.g. wandb api key

- `src/Workflow` — generic protocols: factories, registries, the Hydra config
  layer, runtime-context setup.
- `src/AIML` — generic, domain-agnostic ML pipeline machinery.
- `src/AcVideoJEPA` — the concrete experiment (the JEPA objective, two-rooms
  data, backbones, latent rollout, the Lightning module, and its config groups).

See [`src/README.md`](src/README.md) for the architecture and design rationale.

## Installation

### 1. Clone

    git clone <repo-url> eb_jepa_cleaned
    cd eb_jepa_cleaned

### 2. Create the environment (conda)

A dedicated Python 3.12 environment:

    conda create -n jepa python=3.12 -y
    conda activate jepa

### 3. Install the package

Editable install with the experiment + dev extras (pulls torch, lightning,
hydra-core, the two-rooms deps, pytest, …):

    pip install -e ".[acvideo,dev]"

Extras: `acvideo` (the experiment: gymnasium/scipy/opencv/wandb/…), `dev`
(pytest). The core framework alone is `pip install -e .`.

Check the install:

    python -c "import src.AcVideoJEPA; print('ok')"
    pytest -q

## Launching experiments

Run configs live in [`Configs/`](Configs/) — **one `.yaml` = one run**. Each
reuses the framework's config groups (`src/AcVideoJEPA/configs/`) and sets the
run's identity (`setup.paths.run_name`). Launch with:

    python -m src.AIML.Execution.launch Configs --config-name toy_run

- `--mode {train,resume,validate}` — overrides `run.mode` (default `train`).
- `--ckpt PATH` — checkpoint for `resume` / `validate`.
- `--overwrite` / `--ask-overwrite` — existing-results policy.
- trailing `key=value` — Hydra overrides, e.g. `trainer.max_steps=500`.

Results are written under `runs/<experiment>/<run_name>/` (config snapshot, logs,
checkpoints). Re-running a config whose results already exist asks whether to
delete them first (see [`Configs/README.md`](Configs/README.md)).

To create a new run, copy `Configs/toy_run.yaml`, rename it, set its `run_name`,
and edit the overrides.

## Weights & Biases (optional)

Enable wandb logging by selecting the logger group and the online setup:

    python -m src.AIML.Execution.launch Configs --config-name toy_run \
        loggers=wandb setup.wandb.enabled=true setup.wandb.mode=online setup.wandb.login=true

The API key is needed **once per machine** (cached in `~/.netrc`). Provide it by
running `wandb login`, exporting `WANDB_API_KEY`, or putting it in a gitignored
`user_credential.yaml` (`{wandb: {api_key: ...}}`) and enabling
`setup.user_credential` — the setup step exports it before wandb login. Offline
logging (`loggers=wandb`, default offline) needs no key.
