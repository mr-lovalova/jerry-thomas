from pathlib import Path

import pytest
from pydantic import ValidationError

from jerrythomas.services.scaffold.paths import (
    default_project_yaml_path,
    ensure_project_scaffold,
)


def test_default_project_path_matches_plugin_scaffold(tmp_path: Path) -> None:
    assert default_project_yaml_path(tmp_path) == (
        tmp_path / "your-dataset" / "project.yaml"
    )


def test_ensure_project_scaffold_creates_dotenv_example(tmp_path: Path) -> None:
    project_yaml = tmp_path / "example" / "project.yaml"

    ensure_project_scaffold(project_yaml)

    dotenv_example = project_yaml.parent / ".env.example"
    assert dotenv_example.exists()
    text = dotenv_example.read_text(encoding="utf-8")
    assert "RAW_ROOT=" in text


def test_ensure_project_scaffold_reports_invalid_project(tmp_path: Path) -> None:
    project_yaml = tmp_path / "project.yaml"
    project_yaml.write_text(
        "schema_version: 6\npaths:\n  profiels: ./profiles\n",
        encoding="utf-8",
    )

    with pytest.raises(ValidationError):
        ensure_project_scaffold(project_yaml)
