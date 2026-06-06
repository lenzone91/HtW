# JEPA Templates

Small reusable bricks for HackTheWorld JEPA experiments.

The intended flow for a new world-model experiment is:

1. Make the dataset return a dict with `obs`, `actions`, and optional `probe_target`.
2. Build a model with `build_impala_rnn_ac_jepa`.
3. Call `train_ac_jepa_step` inside a project-specific training loop.
4. Add probes and planning evaluation outside the core world-model step.

Expected batch shapes:

```text
obs:     [B, C, T, H, W]
actions: [B, A, T]
```

For IDM-regularized AC-JEPA training, keep `actions` aligned with `obs` on the
time dimension. If the environment naturally provides transition actions
`[B, A, T - 1]`, pad or repeat one action before calling the shared train step.

The helper `ensure_eb_jepa_importable()` lets these templates import EB-JEPA from
the git submodule without requiring a separate editable install.
