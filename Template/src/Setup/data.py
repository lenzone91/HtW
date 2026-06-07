from pathlib import Path

from ..Utils.error import handle_error
from .paths import resolve_path


def setup_data(
    data_config: dict | None = None,
    project_root: str | Path | None = None,
) -> dict:
    """
    Resolve dataset root paths, validate them, and return data context.
    """
    if data_config is None:
        data_config = {}

    strict = data_config.get("strict", True)
    dataset_roots = data_config.get("dataset_roots", {})

    resolved_dataset_roots = resolve_dataset_roots(
        dataset_roots=dataset_roots,
        project_root=project_root,
    )

    check_dataset_roots_exist(
        dataset_roots=resolved_dataset_roots,
        strict=strict,
    )

    return {
        "dataset_roots": stringify_dataset_roots(resolved_dataset_roots),
    }


def resolve_dataset_roots(
    dataset_roots: dict[str, str | Path],
    project_root: str | Path | None = None,
) -> dict[str, Path]:
    """
    Resolve all named dataset root paths.
    """
    return {
        dataset_name: resolve_path(dataset_root, root=project_root)
        for dataset_name, dataset_root in dataset_roots.items()
    }


def check_dataset_roots_exist(
    dataset_roots: dict[str, Path],
    strict: bool = True,
) -> None:
    """
    Check that all configured dataset roots exist.
    """
    for dataset_name, dataset_root in dataset_roots.items():
        check_dataset_root_exists(
            dataset_name=dataset_name,
            dataset_root=dataset_root,
            strict=strict,
        )


def check_dataset_root_exists(
    dataset_name: str,
    dataset_root: Path,
    strict: bool = True,
) -> None:
    """
    Check that one dataset root exists and is a directory.
    """
    if not dataset_root.exists():
        handle_error(
            msg=f"Dataset root for '{dataset_name}' does not exist: {dataset_root}",
            strict=strict,
        )
        return

    if not dataset_root.is_dir():
        handle_error(
            msg=f"Dataset root for '{dataset_name}' is not a directory: {dataset_root}",
            strict=strict,
        )


def stringify_dataset_roots(
    dataset_roots: dict[str, Path],
) -> dict[str, str]:
    """
    Convert resolved dataset root paths to strings.
    """
    return {
        dataset_name: str(dataset_root)
        for dataset_name, dataset_root in dataset_roots.items()
    }