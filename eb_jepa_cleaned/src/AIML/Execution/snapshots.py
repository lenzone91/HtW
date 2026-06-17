"""
Per-run reproducibility snapshot.

Shared by the run flows: if the runtime context carries a run `configs_dir`, the
resolved config and runtime context are written there. No-op when there is no
run directory (e.g. tests passing `runtime_context=None`).
"""

from ...Workflow.Configs.savings import save_execution_snapshots


def snapshot_execution(config: dict, runtime_context: dict | None) -> None:
    if not runtime_context:
        return
    configs_dir = runtime_context.get("paths", {}).get("configs_dir")
    if configs_dir:
        save_execution_snapshots(config, runtime_context, snapshot_dir=configs_dir)
