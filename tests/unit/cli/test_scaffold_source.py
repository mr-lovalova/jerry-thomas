from pathlib import Path

import pytest
import yaml

from jerrythomas.config.sources import SourceConfig
from jerrythomas.services.scaffold.paths import ensure_project_scaffold
from jerrythomas.services.scaffold.source_yaml import (
    create_source_yaml,
    default_loader_config,
)


def _create_plugin(tmp_path: Path) -> Path:
    root = tmp_path / "sample_plugin"
    pkg_dir = root / "src" / "sample_plugin"
    pkg_dir.mkdir(parents=True)
    (pkg_dir / "__init__.py").write_text("", encoding="utf-8")
    (root / "pyproject.toml").write_text(
        '[project]\nname = "sample-plugin"\nversion = "0.0.0"\n',
        encoding="utf-8",
    )
    return root


def test_create_source_scaffolds_into_default_project(tmp_path: Path) -> None:
    plugin_root = _create_plugin(tmp_path)

    loader = default_loader_config("fs", "csv")
    assert "glob" not in loader
    create_source_yaml(
        source_id="demo.weather",
        loader=loader,
        parser_ep="identity",
        root=plugin_root,
    )

    expected = plugin_root / "your-project" / "sources" / "demo.weather.yaml"
    assert expected.exists(), f"expected scaffolded source at {expected}"
    document = yaml.safe_load(expected.read_text(encoding="utf-8"))
    SourceConfig.model_validate(document)
    assert document["loader"] == loader


def test_create_source_preserves_custom_source_id(tmp_path: Path) -> None:
    plugin_root = _create_plugin(tmp_path)

    create_source_yaml(
        source_id="demo.weather.hourly",
        loader={"entrypoint": "custom.loader", "args": {}},
        parser_ep="identity",
        root=plugin_root,
    )

    path = plugin_root / "your-project" / "sources" / "demo.weather.hourly.yaml"
    assert yaml.safe_load(path.read_text(encoding="utf-8"))["id"] == (
        "demo.weather.hourly"
    )


def test_create_source_quotes_yaml_sensitive_entrypoints(tmp_path: Path) -> None:
    plugin_root = _create_plugin(tmp_path)
    path = create_source_yaml(
        source_id="demo.weather",
        loader={"entrypoint": 'custom"loader\\name', "args": {}},
        parser_ep='custom"parser\\name',
        root=plugin_root,
    )

    source = SourceConfig.model_validate(
        yaml.safe_load(path.read_text(encoding="utf-8"))
    )

    assert source.loader.entrypoint == 'custom"loader\\name'
    assert source.parser.entrypoint == 'custom"parser\\name'


def test_default_loader_config_rejects_pickle_format() -> None:
    with pytest.raises(ValueError, match="Unsupported source format: 'pickle'"):
        default_loader_config("fs", "pickle")


@pytest.mark.parametrize(
    ("transport", "fmt", "reader"),
    [
        (
            "fs",
            "csv",
            {
                "format": "csv",
                "encoding": "utf-8",
                "delimiter": ",",
            },
        ),
        ("fs", "json", {"format": "json", "encoding": "utf-8"}),
        ("fs", "jsonl", {"format": "jsonl", "encoding": "utf-8"}),
        ("fs", "parquet", {"format": "parquet"}),
        (
            "http",
            "csv",
            {
                "format": "csv",
                "encoding": "utf-8",
                "delimiter": ",",
            },
        ),
        ("http", "json", {"format": "json", "encoding": "utf-8"}),
        ("http", "jsonl", {"format": "jsonl", "encoding": "utf-8"}),
    ],
)
def test_default_loader_config_builds_builtin_source(
    transport: str,
    fmt: str,
    reader: dict[str, object],
) -> None:
    loader = default_loader_config(transport, fmt)

    if transport == "fs":
        assert loader == {
            "transport": "fs",
            "path": "<PATH OR GLOB>",
            "reader": reader,
        }
    else:
        assert loader == {
            "transport": "http",
            "url": "<https://api.example.com/data.json>",
            "headers": {},
            "params": {},
            "reader": reader,
        }


def test_default_loader_config_preserves_synthetic_loader() -> None:
    assert default_loader_config("synthetic", None) == {
        "entrypoint": "core.synthetic.ticks",
        "args": {"start": "<ISO8601>", "end": "<ISO8601>", "frequency": "1h"},
    }


def test_default_loader_config_rejects_http_parquet() -> None:
    with pytest.raises(ValueError, match="HTTP sources do not support parquet"):
        default_loader_config("http", "parquet")


def test_default_loader_config_rejects_unknown_transport() -> None:
    with pytest.raises(ValueError, match="Unsupported source transport"):
        default_loader_config("unknown", None)


def test_create_source_refuses_to_replace_existing_config(tmp_path: Path) -> None:
    plugin_root = _create_plugin(tmp_path)
    path = create_source_yaml(
        source_id="demo.weather",
        loader={"entrypoint": "custom.loader", "args": {}},
        parser_ep="identity",
        root=plugin_root,
    )
    original = path.read_bytes()

    with pytest.raises(FileExistsError, match="demo.weather.yaml"):
        create_source_yaml(
            source_id="demo.weather",
            loader={"entrypoint": "different.loader", "args": {"changed": True}},
            parser_ep="custom.parser",
            root=plugin_root,
        )

    assert path.read_bytes() == original


def test_create_source_rejects_existing_id_in_another_root(tmp_path: Path) -> None:
    plugin_root = _create_plugin(tmp_path)
    project_yaml = plugin_root / "your-project" / "project.yaml"
    project = ensure_project_scaffold(project_yaml)
    common_sources = plugin_root / "common" / "sources"
    common_sources.mkdir(parents=True)
    project_yaml.write_text(
        project_yaml.read_text(encoding="utf-8").replace(
            "  sources: ./sources",
            "  sources: [./sources, ../common/sources]",
        ),
        encoding="utf-8",
    )
    (common_sources / "existing.yaml").write_text(
        "id: demo.weather\n"
        "parser: {entrypoint: identity}\n"
        "loader: {entrypoint: identity}\n",
        encoding="utf-8",
    )

    with pytest.raises(FileExistsError, match="Source id 'demo.weather'"):
        create_source_yaml(
            source_id="demo.weather",
            loader={"entrypoint": "custom.loader", "args": {}},
            parser_ep="identity",
            root=plugin_root,
            project_yaml=project_yaml,
        )

    assert not (project.source_dirs[0] / "demo.weather.yaml").exists()


def test_create_source_does_not_validate_unrelated_source_values(
    tmp_path: Path,
) -> None:
    plugin_root = _create_plugin(tmp_path)
    project_yaml = plugin_root / "your-project" / "project.yaml"
    project = ensure_project_scaffold(project_yaml)
    (project_yaml.parent / ".env").write_text(
        "SECRET=TOP-SECRET-VALUE\n",
        encoding="utf-8",
    )
    (project.source_dirs[0] / "existing.yaml").write_text(
        "id: demo.existing\n"
        "parser: {entrypoint: identity}\n"
        "loader:\n"
        "  transport: ${env:SECRET}\n"
        "  path: data.jsonl\n"
        "  reader: {format: jsonl}\n",
        encoding="utf-8",
    )

    path = create_source_yaml(
        source_id="demo.weather",
        loader={"entrypoint": "custom.loader", "args": {}},
        parser_ep="identity",
        root=plugin_root,
        project_yaml=project_yaml,
    )

    assert path.name == "demo.weather.yaml"


@pytest.mark.parametrize(
    "source_id",
    ["weather", "demo/weather", "demo..weather", "demo.café"],
)
def test_create_source_rejects_unsafe_source_id(
    source_id: str,
    tmp_path: Path,
) -> None:
    plugin_root = _create_plugin(tmp_path)

    with pytest.raises(ValueError, match="source_id"):
        create_source_yaml(
            source_id=source_id,
            loader={"entrypoint": "custom.loader", "args": {}},
            parser_ep="identity",
            root=plugin_root,
        )
