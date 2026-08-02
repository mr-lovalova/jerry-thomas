import textwrap
from pathlib import Path

import pytest
from pydantic import ValidationError

from jerrythomas.cli.workspace import load_workspace_context


def _write_jerry(tmp_path: Path, content: str) -> Path:
    path = tmp_path / "jerry.yaml"
    path.write_text(textwrap.dedent(content), encoding="utf-8")
    return path


def test_workspace_rejects_shared_block(tmp_path):
    _write_jerry(
        tmp_path,
        """
        shared:
          observability:
            visuals: OFF
        """,
    )

    with pytest.raises(ValidationError, match="shared"):
        load_workspace_context(tmp_path)


def test_workspace_rejects_serve_defaults_block(tmp_path):
    _write_jerry(
        tmp_path,
        """
        serve:
          limit: 25
        """,
    )

    with pytest.raises(ValidationError, match="serve"):
        load_workspace_context(tmp_path)


def test_workspace_rejects_build_defaults_block(tmp_path):
    _write_jerry(
        tmp_path,
        """
        build:
          mode: AUTO
        """,
    )

    with pytest.raises(ValidationError, match="build"):
        load_workspace_context(tmp_path)


def test_workspace_rejects_serve_observability_block(tmp_path):
    _write_jerry(
        tmp_path,
        """
        serve:
          observability:
            logging:
              outputs:
                - transport: fs
                  scope: execution
        """,
    )

    with pytest.raises(ValidationError, match="serve"):
        load_workspace_context(tmp_path)


def test_workspace_resolve_dataset_alias(tmp_path: Path):
    _write_jerry(
        tmp_path,
        """
        datasets:
          example: example/project.yaml
        default_dataset: example
        """,
    )
    (tmp_path / "example").mkdir(parents=True, exist_ok=True)
    (tmp_path / "example" / "project.yaml").write_text(
        "schema_version: 5\nartifact_revision: 1\nname: x\npaths:\n  streams: ./streams\n  sources: ./sources\n  dataset: dataset.yaml\n  artifacts: ./artifacts\n  operations: ./operations\n",
        encoding="utf-8",
    )

    context = load_workspace_context(tmp_path)
    assert context
    resolved = context.resolve_dataset_alias("example")
    assert resolved is not None
    assert resolved.name == "project.yaml"
