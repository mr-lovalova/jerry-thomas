from pathlib import Path

from datapipeline.artifacts.registry import ArtifactRecord
from datapipeline.services.path_policy import (
    resolve_project_path,
    resolve_workspace_path,
)


def test_resolve_workspace_path_prefers_workspace_root(tmp_path: Path) -> None:
    workspace_root = tmp_path / "workspace"
    workspace_root.mkdir()
    nested = workspace_root / "nested" / "cwd"
    nested.mkdir(parents=True)

    resolved = resolve_workspace_path("data/out.jsonl", workspace_root, cwd=nested)
    assert resolved == (workspace_root / "data" / "out.jsonl").resolve()


def test_resolve_project_path_uses_project_directory(tmp_path: Path) -> None:
    project_dir = tmp_path / "plugin" / "dataset"
    project_dir.mkdir(parents=True)
    project_yaml = project_dir / "project.yaml"
    project_yaml.write_text("schema_version: 4\nname: x\npaths: {}\n", encoding="utf-8")

    assert (
        resolve_project_path(project_yaml, "sources").resolve()
        == (project_dir / "sources").resolve()
    )
    absolute = tmp_path / "absolute" / "target"
    assert resolve_project_path(project_yaml, absolute) == absolute.resolve()


def test_artifact_record_resolve_uses_relative_policy(tmp_path: Path) -> None:
    root = tmp_path / "artifacts"
    rec = ArtifactRecord(relative_path="schema.json", meta={})
    assert rec.resolve(root) == root / "schema.json"
