from pathlib import Path
import textwrap

import pytest
import yaml

from jerrythomas.cli.commands.stream import handle as handle_stream_create
from jerrythomas.config.streams import (
    CombinedStreamConfig,
    SourceStreamConfig,
)
from jerrythomas.plugins import MAPPERS_EP
from jerrythomas.services.project import load_project
from jerrythomas.services.streams.loader import load_streams


def _create_plugin(tmp_path: Path) -> Path:
    root = tmp_path / "plugin"
    pkg_dir = root / "src" / "sample_plugin"
    pkg_dir.mkdir(parents=True)
    (pkg_dir / "__init__.py").write_text("", encoding="utf-8")
    pyproject = textwrap.dedent(
        """
        [project]
        name = "sample-plugin"
        version = "0.1.0"
        """
    ).strip()
    (root / "pyproject.toml").write_text(pyproject + "\n", encoding="utf-8")

    return root


def _write_project_yaml(
    project_path: Path, sources_dir: Path, streams_dir: Path
) -> None:
    project_path.parent.mkdir(parents=True, exist_ok=True)
    sources_dir.mkdir(parents=True, exist_ok=True)
    streams_dir.mkdir(parents=True, exist_ok=True)
    content = textwrap.dedent(
        f"""
        schema_version: 7
        artifact_revision: 1
        paths:
          streams: {streams_dir}
          sources: {sources_dir}
          datasets: datasets
          artifacts: {streams_dir.parent / "build"}
        globals: {{}}
        """
    ).strip()
    project_path.write_text(content + "\n", encoding="utf-8")


def _write_source_yaml(path: Path, alias: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    content = textwrap.dedent(
        f"""
        id: {alias}
        parser:
          entrypoint: core.identity
          args: {{}}
        loader:
          entrypoint: identity
          args: {{}}
        freshness: opaque
        """
    ).strip()
    path.write_text(content + "\n", encoding="utf-8")


def _write_source_stream_yaml(path: Path, stream_id: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        textwrap.dedent(
            f"""
            id: {stream_id}
            from:
              source: demo.weather
            map:
              entrypoint: core.identity
              args: {{}}
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )


def _input_sequence(monkeypatch: pytest.MonkeyPatch, responses: list[str]) -> None:
    iterator = iter(responses)

    def _fake_input(_: str = "") -> str:
        return next(iterator)

    monkeypatch.setattr("builtins.input", _fake_input)


def test_stream_scaffold_uses_project_paths(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Stream scaffold should use the project's streams/sources paths inside the plugin repo."""
    plugin_root = _create_plugin(tmp_path)
    # Create a project with explicit streams and sources paths.
    dataset_root = plugin_root / "your-project"
    sources_dir = dataset_root / "sources"
    streams_dir = dataset_root / "streams"
    project_yaml = dataset_root / "project.yaml"
    _write_project_yaml(project_yaml, sources_dir, streams_dir)
    _write_source_yaml(sources_dir / "demo.weather.yaml", "demo.weather")

    _input_sequence(monkeypatch, ["1", "1", "1", ""])

    handle_stream_create(plugin_root=plugin_root)

    stream_path = streams_dir / "weather.weather.yaml"
    assert stream_path.exists(), f"expected stream at {stream_path}"
    streams = load_streams(load_project(project_yaml))
    stream = streams.streams["weather.weather"]
    assert isinstance(stream, SourceStreamConfig)
    assert stream.map.entrypoint == "core.identity"
    assert stream.preprocess == []
    package = plugin_root / "src" / "sample_plugin"
    assert not (package / "dtos").exists()
    assert not (package / "domains").exists()
    assert not (package / "mappers").exists()


def test_stream_identity_mapper_skips_scaffold(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    plugin_root = _create_plugin(tmp_path)
    dataset_root = plugin_root / "your-project"
    sources_dir = dataset_root / "sources"
    streams_dir = dataset_root / "streams"
    _write_project_yaml(dataset_root / "project.yaml", sources_dir, streams_dir)
    _write_source_yaml(sources_dir / "demo.weather.yaml", "demo.weather")

    _input_sequence(monkeypatch, ["1", "1", ""])

    handle_stream_create(plugin_root=plugin_root, use_identity=True)

    stream_path = streams_dir / "weather.weather.yaml"
    assert stream_path.exists()
    document = yaml.safe_load(stream_path.read_text(encoding="utf-8"))
    assert document["map"]["entrypoint"] == "core.identity"
    package = plugin_root / "src" / "sample_plugin"
    assert not (package / "dtos").exists()
    assert not (package / "domains").exists()
    assert not (package / "mappers").exists()


def test_source_stream_name_abort_does_not_write_config(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    plugin_root = _create_plugin(tmp_path)
    dataset_root = plugin_root / "your-project"
    sources_dir = dataset_root / "sources"
    streams_dir = dataset_root / "streams"
    _write_project_yaml(dataset_root / "project.yaml", sources_dir, streams_dir)
    _write_source_yaml(sources_dir / "demo.weather.yaml", "demo.weather")
    pyproject = plugin_root / "pyproject.toml"
    original_pyproject = pyproject.read_bytes()

    monkeypatch.setattr(
        "jerrythomas.cli.commands.stream.pick_from_menu",
        lambda *args, **kwargs: "source",
    )
    monkeypatch.setattr(
        "jerrythomas.cli.commands.stream.pick_from_list",
        lambda *args, **kwargs: "demo.weather",
    )
    monkeypatch.setattr(
        "jerrythomas.cli.commands.stream._select_source_mapper",
        lambda root: "custom.mapper",
    )

    def abort_stream_name(prompt: str, default: str | None = None) -> str:
        raise RuntimeError("prompt aborted")

    monkeypatch.setattr(
        "jerrythomas.cli.commands.stream.choose_name",
        abort_stream_name,
    )

    with pytest.raises(RuntimeError, match="prompt aborted"):
        handle_stream_create(plugin_root=plugin_root)

    package = plugin_root / "src" / "sample_plugin"
    assert not (package / "dtos").exists()
    assert not (package / "domains").exists()
    assert not (package / "mappers").exists()
    assert not (streams_dir / "weather.weather.yaml").exists()
    assert pyproject.read_bytes() == original_pyproject


def test_source_stream_preserves_source_variant_in_default_stream_id(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    plugin_root = _create_plugin(tmp_path)
    dataset_root = plugin_root / "your-project"
    sources_dir = dataset_root / "sources"
    streams_dir = dataset_root / "streams"
    _write_project_yaml(dataset_root / "project.yaml", sources_dir, streams_dir)
    _write_source_yaml(
        sources_dir / "demo.weather.benchmark.yaml",
        "demo.weather.benchmark",
    )

    _input_sequence(monkeypatch, ["1", "1", ""])

    handle_stream_create(plugin_root=plugin_root, use_identity=True)

    assert (streams_dir / "weather.weather.benchmark.yaml").exists()


def test_source_stream_selects_existing_mapper_reference(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    plugin_root = _create_plugin(tmp_path)
    pyproject = plugin_root / "pyproject.toml"
    with pyproject.open("a", encoding="utf-8") as project_file:
        project_file.write(
            f'\n[project.entry-points."{MAPPERS_EP}"]\n'
            'weather = "sample_plugin.mappers.weather:map_weather"\n'
        )
    dataset_root = plugin_root / "your-project"
    sources_dir = dataset_root / "sources"
    streams_dir = dataset_root / "streams"
    _write_project_yaml(dataset_root / "project.yaml", sources_dir, streams_dir)
    _write_source_yaml(sources_dir / "demo.weather.yaml", "demo.weather")

    _input_sequence(monkeypatch, ["1", "1", "1", "1", ""])

    handle_stream_create(plugin_root=plugin_root)

    config = yaml.safe_load(
        (streams_dir / "weather.weather.yaml").read_text(encoding="utf-8")
    )
    assert config["map"]["entrypoint"] == "weather"
    package = plugin_root / "src" / "sample_plugin"
    assert not (package / "dtos").exists()
    assert not (package / "domains").exists()
    assert not (package / "mappers").exists()


def test_source_stream_accepts_custom_mapper_reference(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    plugin_root = _create_plugin(tmp_path)
    dataset_root = plugin_root / "your-project"
    sources_dir = dataset_root / "sources"
    streams_dir = dataset_root / "streams"
    _write_project_yaml(dataset_root / "project.yaml", sources_dir, streams_dir)
    _write_source_yaml(sources_dir / "demo.weather.yaml", "demo.weather")

    _input_sequence(monkeypatch, ["1", "1", "2", "custom.mapper", ""])

    handle_stream_create(plugin_root=plugin_root)

    config = yaml.safe_load(
        (streams_dir / "weather.weather.yaml").read_text(encoding="utf-8")
    )
    assert config["map"]["entrypoint"] == "custom.mapper"
    package = plugin_root / "src" / "sample_plugin"
    assert not (package / "dtos").exists()
    assert not (package / "domains").exists()
    assert not (package / "mappers").exists()


@pytest.mark.parametrize("stream_choice", ["2", "3"])
def test_stream_identity_flag_rejects_non_source_stream(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    stream_choice: str,
) -> None:
    plugin_root = _create_plugin(tmp_path)
    _input_sequence(monkeypatch, [stream_choice])
    with pytest.raises(SystemExit):
        handle_stream_create(plugin_root=plugin_root, use_identity=True)


def test_aligned_stream_scaffold_writes_ordered_inputs(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    plugin_root = _create_plugin(tmp_path)
    dataset_root = plugin_root / "your-project"
    sources_dir = dataset_root / "sources"
    streams_dir = dataset_root / "streams"
    _write_project_yaml(dataset_root / "project.yaml", sources_dir, streams_dir)
    _write_source_yaml(sources_dir / "demo.weather.yaml", "demo.weather")
    _write_source_stream_yaml(streams_dir / "weather.pressure.yaml", "weather.pressure")
    _write_source_stream_yaml(
        streams_dir / "weather.temperature.yaml",
        "weather.temperature",
    )

    _input_sequence(
        monkeypatch,
        [
            "2",  # aligned stream
            "1,2",  # input stream order
            "air_density.processed",
            "combine_air_density",
        ],
    )

    handle_stream_create(plugin_root=plugin_root)

    stream_path = streams_dir / "air_density.processed.yaml"
    assert stream_path.exists()
    config = yaml.safe_load(stream_path.read_text(encoding="utf-8"))
    spec = CombinedStreamConfig.model_validate(config)
    assert spec.from_.stream == "weather.pressure"
    assert list(spec.join.partner_stream_ids()) == ["weather.temperature"]
    assert spec.combine.entrypoint == "combine_air_density"
    assert spec.combine.args == {}


def test_aligned_stream_scaffold_selects_registered_combiner(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    plugin_root = _create_plugin(tmp_path)
    pyproject = plugin_root / "pyproject.toml"
    pyproject.write_text(
        pyproject.read_text(encoding="utf-8")
        + '\n[project.entry-points."jerrythomas.combiners"]\n'
        + 'air_density = "sample_plugin.combiners:air_density"\n',
        encoding="utf-8",
    )
    dataset_root = plugin_root / "your-project"
    sources_dir = dataset_root / "sources"
    streams_dir = dataset_root / "streams"
    _write_project_yaml(dataset_root / "project.yaml", sources_dir, streams_dir)
    _write_source_yaml(sources_dir / "demo.weather.yaml", "demo.weather")
    _write_source_stream_yaml(streams_dir / "weather.pressure.yaml", "weather.pressure")
    _write_source_stream_yaml(
        streams_dir / "weather.temperature.yaml",
        "weather.temperature",
    )
    _input_sequence(
        monkeypatch,
        ["2", "1,2", "air_density.processed", "1", "1"],
    )

    handle_stream_create(plugin_root=plugin_root)

    config = yaml.safe_load(
        (streams_dir / "air_density.processed.yaml").read_text(encoding="utf-8")
    )
    assert config["combine"]["entrypoint"] == "air_density"


def test_broadcast_stream_scaffold_selects_partitioned_and_global_inputs(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    plugin_root = _create_plugin(tmp_path)
    dataset_root = plugin_root / "your-project"
    sources_dir = dataset_root / "sources"
    streams_dir = dataset_root / "streams"
    project_yaml = dataset_root / "project.yaml"
    _write_project_yaml(project_yaml, sources_dir, streams_dir)
    _write_source_yaml(sources_dir / "demo.weather.yaml", "demo.weather")
    primary_path = streams_dir / "weather.temperature.yaml"
    _write_source_stream_yaml(primary_path, "weather.temperature")
    primary_path.write_text(
        primary_path.read_text(encoding="utf-8") + "partition_by: [station]\n",
        encoding="utf-8",
    )
    _write_source_stream_yaml(
        streams_dir / "weather.baseline.yaml",
        "weather.baseline",
    )

    _input_sequence(
        monkeypatch,
        [
            "3",  # broadcast stream
            "1",  # partitioned primary
            "1",  # global broadcast input
            "weather.adjusted",
            "attach_baseline",
        ],
    )

    handle_stream_create(plugin_root=plugin_root)

    stream_path = streams_dir / "weather.adjusted.yaml"
    config = CombinedStreamConfig.model_validate(
        yaml.safe_load(stream_path.read_text(encoding="utf-8"))
    )
    assert config.from_.stream == "weather.temperature"
    assert config.join.with_ == "weather.baseline"
    assert config.combine.entrypoint == "attach_baseline"
