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
    project_path = plugin_root / "your-project" / "project.yaml"
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

    with pytest.raises(SystemExit, match="your-project/project.yaml"):
        handle("sources", plugin_root=plugin_root)


def _catalog_project(tmp_path: Path) -> Path:
    for directory in ("sources", "streams", "datasets", "operations", "profiles"):
        (tmp_path / directory).mkdir()
    project = tmp_path / "project.yaml"
    project.write_text(
        "schema_version: 7\nartifact_revision: 1\npaths:\n"
        "  sources: sources\n  streams: streams\n  datasets: datasets\n"
        "  operations: operations\n  profiles: profiles\n  artifacts: artifacts\n",
        encoding="utf-8",
    )
    for name, version in (("daily", "v2"), ("monthly", "v3")):
        (tmp_path / "datasets" / f"{name}.yaml").write_text(
            f"version: {version}\nsample: {{cadence: 1d, rounding: exact}}\n",
            encoding="utf-8",
        )
        (tmp_path / "operations" / f"{name}.yaml").write_text(
            f"kind: output\nentrypoint: core.dataset\ndataset: {name}\n",
            encoding="utf-8",
        )
        (tmp_path / "profiles" / f"serve.{name}.yaml").write_text(
            f"operation: {name}\nenabled: {'true' if name == 'daily' else 'false'}\n",
            encoding="utf-8",
        )
    return project


def test_list_datasets_reports_named_catalog_versions(tmp_path, capsys) -> None:
    project = _catalog_project(tmp_path)

    handle("datasets", project=str(project))

    assert capsys.readouterr().out.splitlines() == [
        "daily\tversion=v2",
        "monthly\tversion=v3",
    ]


def test_list_profiles_reports_binding_and_disabled_profiles(tmp_path, capsys) -> None:
    project = _catalog_project(tmp_path)

    handle("profiles", project=str(project))

    assert capsys.readouterr().out.splitlines() == [
        "serve.daily\toperation=daily\tdataset=daily\tenabled=true",
        "serve.monthly\toperation=monthly\tdataset=monthly\tenabled=false",
    ]


def test_list_streams_uses_explicit_project(monkeypatch, tmp_path, capsys) -> None:
    project = tmp_path / "project.yaml"
    monkeypatch.setattr(
        "jerrythomas.cli.commands.list_.load_project", lambda path: path
    )
    monkeypatch.setattr(
        "jerrythomas.cli.commands.list_.load_streams",
        lambda path: SimpleNamespace(
            streams={"weather.daily": object(), "weather.hourly": object()}
        ),
    )

    handle("streams", project=str(project))

    assert capsys.readouterr().out.splitlines() == ["weather.daily", "weather.hourly"]
