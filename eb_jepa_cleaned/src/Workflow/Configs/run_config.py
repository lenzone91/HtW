"""
Run-config resolution.

A run is authored as a *folder* of YAML fragments — `setup.yaml`,
`datamodule.yaml`, `module.yaml`, `trainer.yaml`, `loggers.yaml`, `run.yaml`, …
— with a `config.yaml` entry point whose `defaults:` list composes them. The
composition itself is done by **Hydra** (`load_resolved_config`); this module is
the thin custom entry point over it (resolve a folder or a resolved file, and
optionally write the merged sibling snapshot).

    Configs/
      run_1/            <- fragments + config.yaml entry  (editable source)
        config.yaml
        setup.yaml
        ...
      run_1.yaml        <- merged/composed snapshot        (generated; reproducibility)

`resolve_run_config` accepts either form and returns a resolved plain dict.
"""

import argparse
from pathlib import Path

from .compose import load_resolved_config
from .conversion import save_config
from .errors import ConfigError

# The entry-point config name inside a run folder.
RUN_ENTRY_NAME = "config"


def resolve_run_config(config_path, overrides: list[str] | None = None) -> dict:
    """
    Resolve a run config given as a fragment folder (composes its `config` entry)
    or as a single resolved YAML file. Composition is performed by Hydra.
    """
    path = Path(config_path).expanduser().resolve()

    if path.is_dir():
        return load_resolved_config(path, RUN_ENTRY_NAME, overrides)

    if path.is_file():
        # A resolved snapshot (or any single entry yaml): compose it in place.
        return load_resolved_config(path.parent, path.stem, overrides)

    raise ConfigError(f"Run config path does not exist: {path}")


def save_composed_run(
    config_path,
    overrides: list[str] | None = None,
    output_path=None,
) -> Path:
    """
    Compose a run folder and write the merged snapshot next to it
    (`Configs/run_1/` -> `Configs/run_1.yaml`). Returns the snapshot path.
    """
    config = resolve_run_config(config_path, overrides)
    path = Path(config_path).expanduser().resolve()
    if output_path is None:
        output_path = path.with_suffix(".yaml")
    save_config(config, output_path)
    return Path(output_path)


def main(argv: list[str] | None = None) -> Path:
    parser = argparse.ArgumentParser(
        description="Compose a run-config folder into its merged snapshot."
    )
    parser.add_argument("run_path", help="Run folder (or a resolved .yaml).")
    parser.add_argument("-o", "--output", default=None, help="Snapshot output path.")
    parser.add_argument("overrides", nargs="*", help="Hydra-style overrides.")
    args = parser.parse_args(argv)

    snapshot = save_composed_run(
        args.run_path, overrides=args.overrides, output_path=args.output
    )
    print(f"Wrote composed run config: {snapshot}")
    return snapshot


if __name__ == "__main__":
    main()
