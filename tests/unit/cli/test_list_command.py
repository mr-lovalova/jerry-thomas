from pathlib import Path
from types import SimpleNamespace

import pytest

from jerrythomas.cli.commands.list_ import handle


def test_list_sources_uses_standard_dataset_project(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    plugin_root = tmp_path / "plugin"
    project_path = plugin_root / "your-dataset" / "project.yaml"
    project_path.parent.mkdir(parents=True)
    project_path.touch()
    monkeypatch.setattr(
        "jerrythomas.cli.commands.list_.pkg_root",
        lambda _: (plugin_root, "sample_plugin", plugin_root / "pyproject.toml"),
    )

    def load_project(path: Path) -> Path:
        assert path == project_path
        return path

    monkeypatch.setattr(
        "jerrythomas.cli.commands.list_.load_project",
        load_project,
    )
    monkeypatch.setattr(
        "jerrythomas.cli.commands.list_.load_streams",
        lambda path: SimpleNamespace(sources={"weather": object()}),
    )

    handle("sources", plugin_root=plugin_root)

    assert capsys.readouterr().out == "weather\n"


def test_list_sources_reports_standard_project_path(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    plugin_root = tmp_path / "plugin"
    monkeypatch.setattr(
        "jerrythomas.cli.commands.list_.pkg_root",
        lambda _: (plugin_root, "sample_plugin", plugin_root / "pyproject.toml"),
    )

    with pytest.raises(SystemExit, match="your-dataset/project.yaml"):
        handle("sources", plugin_root=plugin_root)
