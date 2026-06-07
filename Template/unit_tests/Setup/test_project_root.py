from pathlib import Path

from Project_1_SepFormer_TSE_HF.idea_1_project_setup.src.Setup.project_root import (
    infer_project_root,
    resolve_project_root,
)
from Project_1_SepFormer_TSE_HF.idea_1_project_setup.src.Setup.paths import (
    resolve_path,
)
from Project_1_SepFormer_TSE_HF.idea_1_project_setup.src.Setup.user_credential import (
    resolve_credential_path,
)


def make_minimal_project(tmp_path: Path) -> Path:
    project_root = tmp_path / "Project" / "idea_1"
    (project_root / "src").mkdir(parents=True)
    (project_root / "configs" / "runs" / "toy_run").mkdir(parents=True)

    return project_root


def test_infer_project_root_from_elementary_config_folder(tmp_path: Path) -> None:
    project_root = make_minimal_project(tmp_path)
    config_path = project_root / "configs" / "runs" / "toy_run"

    assert infer_project_root(config_path) == project_root.resolve()


def test_infer_project_root_from_resolved_config_file(tmp_path: Path) -> None:
    project_root = make_minimal_project(tmp_path)
    config_path = project_root / "configs" / "runs" / "toy_run.yaml"
    config_path.touch()

    assert infer_project_root(config_path) == project_root.resolve()


def test_resolve_project_root_uses_explicit_root_when_provided(tmp_path: Path) -> None:
    project_root = make_minimal_project(tmp_path)

    assert resolve_project_root(project_root=project_root) == project_root.resolve()


def test_resolve_path_resolves_relative_path_from_root(tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir()

    resolved_path = resolve_path("data/toy", root=root)

    assert resolved_path == (root / "data" / "toy").resolve()


def test_resolve_path_preserves_absolute_path(tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir()

    absolute_path = tmp_path / "external" / "data"

    assert resolve_path(absolute_path, root=root) == absolute_path.resolve()


def test_resolve_credential_path_is_project_root_relative(tmp_path: Path) -> None:
    project_root = make_minimal_project(tmp_path)

    credential_path = resolve_credential_path(
        path="user_credential.yaml",
        project_root=project_root,
    )

    assert credential_path == (project_root / "user_credential.yaml").resolve()