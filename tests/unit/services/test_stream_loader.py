import gzip
from pathlib import Path

import pytest

from jerrythomas.config.streams import (
    CombinedStreamConfig,
    DerivedStreamConfig,
    SourceStreamConfig,
)
from jerrythomas.services.project import load_project
from jerrythomas.services.streams.loader import load_streams
from jerrythomas.services.streams.source import build_source


def _sources(project_yaml: Path):
    return load_streams(load_project(project_yaml)).sources


def _write_project_yaml(project_root: Path) -> Path:
    (project_root / "streams").mkdir(parents=True, exist_ok=True)
    project_yaml = project_root / "project.yaml"
    project_yaml.write_text(
        "\n".join(
            [
                "schema_version: 7",
                "artifact_revision: 1",
                "name: sample",
                "paths:",
                "  streams: streams",
                "  sources: sources",
                "  artifacts: build",
                "  operations: operations",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return project_yaml


def _write_source_yaml(
    sources_dir: Path,
    path_value: str,
    compression: str | None = None,
) -> None:
    sources_dir.mkdir(parents=True, exist_ok=True)
    lines = [
        "id: sample.fs",
        "parser:",
        "  entrypoint: core.identity",
        "  args: {}",
        "loader:",
        "  transport: fs",
        f"  path: {path_value}",
        "  reader:",
        "    format: jsonl",
    ]
    if compression is not None:
        lines.append(f"  compression: {compression}")
    (sources_dir / "sample.yaml").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def _write_named_source_yaml(sources_dir: Path, filename: str, source_id: str) -> None:
    sources_dir.mkdir(parents=True, exist_ok=True)
    (sources_dir / filename).write_text(
        "\n".join(
            [
                f"id: {source_id}",
                "parser:",
                "  entrypoint: core.identity",
                "  args: {}",
                "loader:",
                "  transport: fs",
                "  path: data/rows.jsonl",
                "  reader:",
                "    format: jsonl",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def test_load_sources_resolves_fs_path_from_project_root(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    (project_root / "data").mkdir(parents=True)
    (project_root / "data" / "rows.jsonl").write_text("{}", encoding="utf-8")
    (project_root / "streams").mkdir()
    (project_root / "tasks").mkdir()
    (project_root / "build").mkdir()
    (project_root / "dataset.yaml").write_text(
        "sample:\n  rounding: ceil\n  cadence: 1h\nfeatures: []\n",
        encoding="utf-8",
    )
    _write_source_yaml(project_root / "sources", "data/*.jsonl")
    project_yaml = _write_project_yaml(project_root)

    loaded = _sources(project_yaml)
    source = build_source(
        loaded["sample.fs"],
        project_yaml=project_yaml,
    )
    path_value = source.loader.transport.pattern
    assert path_value == str((project_root / "data" / "*.jsonl").resolve())


def test_load_source_passes_explicit_gzip_compression_to_builtin_loader(
    tmp_path: Path,
) -> None:
    project_root = tmp_path / "project"
    data_dir = project_root / "data"
    data_dir.mkdir(parents=True)
    for index in (1, 2):
        with gzip.open(
            data_dir / f"{index}.jsonl.gz",
            "wt",
            encoding="utf-8",
        ) as stream:
            stream.write(f'{{"value": {index}}}\n')
    _write_source_yaml(
        project_root / "sources",
        "data/*.jsonl.gz",
        compression="gzip",
    )
    project_yaml = _write_project_yaml(project_root)

    source = build_source(
        _sources(project_yaml)["sample.fs"],
        project_yaml=project_yaml,
    )

    assert list(source.stream()) == [{"value": 1}, {"value": 2}]


def test_load_sources_resolves_fs_path_relative_to_project_root_only(
    tmp_path: Path,
) -> None:
    project_root = tmp_path / "workspace" / "demo" / "demo"
    (project_root / "data").mkdir(parents=True)
    (project_root / "data" / "rows.jsonl").write_text("{}", encoding="utf-8")
    resolved_data = project_root / "demo" / "demo" / "data"
    resolved_data.mkdir(parents=True)
    (resolved_data / "rows.jsonl").write_text("{}", encoding="utf-8")
    (project_root / "streams").mkdir()
    (project_root / "tasks").mkdir()
    (project_root / "build").mkdir()
    (project_root / "dataset.yaml").write_text(
        "sample:\n  rounding: ceil\n  cadence: 1h\nfeatures: []\n", encoding="utf-8"
    )
    _write_source_yaml(project_root / "sources", "demo/demo/data/*.jsonl")
    project_yaml = _write_project_yaml(project_root)

    loaded = _sources(project_yaml)
    source = build_source(
        loaded["sample.fs"],
        project_yaml=project_yaml,
    )
    path_value = source.loader.transport.pattern
    assert path_value == str(
        (project_root / "demo" / "demo" / "data" / "*.jsonl").resolve()
    )


def test_load_sources_reads_multiple_source_roots(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    project_root = workspace / "project"
    common_root = workspace / "common"
    (project_root / "streams").mkdir(parents=True)
    (project_root / "tasks").mkdir()
    (project_root / "build").mkdir()
    (project_root / "data").mkdir()
    (project_root / "data" / "rows.jsonl").write_text("{}", encoding="utf-8")
    (project_root / "dataset.yaml").write_text(
        "sample:\n  rounding: ceil\n  cadence: 1h\nfeatures: []\n",
        encoding="utf-8",
    )
    _write_named_source_yaml(project_root / "sources", "local.yaml", "local.fs")
    _write_named_source_yaml(common_root / "sources", "common.yaml", "common.fs")
    project_yaml = project_root / "project.yaml"
    project_yaml.write_text(
        "\n".join(
            [
                "schema_version: 7",
                "artifact_revision: 1",
                "name: sample",
                "paths:",
                "  streams: streams",
                "  sources:",
                "    - sources",
                "    - ../common/sources",
                "  artifacts: build",
                "  operations: operations",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    loaded = _sources(project_yaml)

    assert sorted(loaded) == ["common.fs", "local.fs"]


def test_load_sources_rejects_duplicate_source_ids_across_roots(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    project_root = workspace / "project"
    common_root = workspace / "common"
    (project_root / "streams").mkdir(parents=True)
    (project_root / "tasks").mkdir()
    (project_root / "build").mkdir()
    (project_root / "data").mkdir()
    (project_root / "data" / "rows.jsonl").write_text("{}", encoding="utf-8")
    (project_root / "dataset.yaml").write_text(
        "sample:\n  rounding: ceil\n  cadence: 1h\nfeatures: []\n",
        encoding="utf-8",
    )
    _write_named_source_yaml(project_root / "sources", "local.yaml", "same.fs")
    _write_named_source_yaml(
        common_root / "sources",
        "common.yaml",
        '" same.fs "',
    )
    project_yaml = project_root / "project.yaml"
    project_yaml.write_text(
        "\n".join(
            [
                "schema_version: 7",
                "artifact_revision: 1",
                "name: sample",
                "paths:",
                "  streams: streams",
                "  sources:",
                "    - sources",
                "    - ../common/sources",
                "  artifacts: build",
                "  operations: operations",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Duplicate source id 'same.fs'"):
        _sources(project_yaml)


def test_load_sources_keys_by_interpolated_id(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    _write_named_source_yaml(
        project_root / "sources",
        "source.yaml",
        "${configured_id}",
    )
    project_yaml = _write_project_yaml(project_root)
    project_yaml.write_text(
        project_yaml.read_text(encoding="utf-8")
        + 'globals:\n  configured_id: " canonical.source "\n',
        encoding="utf-8",
    )

    loaded = _sources(project_yaml)

    assert list(loaded) == ["canonical.source"]
    assert loaded["canonical.source"].id == "canonical.source"


def test_load_sources_rejects_non_mapping_yaml(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    sources_dir = project_root / "sources"
    sources_dir.mkdir(parents=True)
    (sources_dir / "bad.yaml").write_text(
        "- not\n- a\n- mapping\n",
        encoding="utf-8",
    )
    project_yaml = _write_project_yaml(project_root)

    with pytest.raises(TypeError, match="Top-level YAML .* must be a mapping"):
        _sources(project_yaml)


def test_load_sources_rejects_missing_id(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    sources_dir = project_root / "sources"
    sources_dir.mkdir(parents=True)
    (sources_dir / "bad.yaml").write_text(
        "parser: {entrypoint: core.identity}\n"
        "loader: {transport: fs, path: data.jsonl, reader: {format: jsonl}}\n",
        encoding="utf-8",
    )
    project_yaml = _write_project_yaml(project_root)

    with pytest.raises(ValueError, match="Missing 'id' in source file"):
        _sources(project_yaml)


def test_load_streams_reads_all_stream_variants_from_one_catalog(
    tmp_path: Path,
) -> None:
    project_root = tmp_path / "project"
    _write_source_yaml(project_root / "sources", "data/*.jsonl")
    streams = project_root / "streams"
    streams.mkdir(parents=True)
    (streams / "prices.yaml").write_text(
        "\n".join(
            [
                "id: prices",
                "from: {source: sample.fs}",
                "map: {entrypoint: map_price}",
                "partition_by: [ticker]",
                "transforms:",
                "  - {operation: dedupe}",
            ]
        ),
        encoding="utf-8",
    )
    (streams / "returns.yaml").write_text(
        "\n".join(
            [
                "id: returns",
                "from: {stream: prices}",
                "transforms:",
                "  - {operation: lag, field: close, periods: 1}",
            ]
        ),
        encoding="utf-8",
    )
    (streams / "aligned.yaml").write_text(
        "\n".join(
            [
                "id: aligned",
                "from: {stream: prices}",
                "join: {kind: align, streams: [returns]}",
                "combine: {entrypoint: combine_records}",
            ]
        ),
        encoding="utf-8",
    )

    loaded = load_streams(load_project(_write_project_yaml(project_root))).streams

    assert isinstance(loaded["prices"], SourceStreamConfig)
    assert isinstance(loaded["returns"], DerivedStreamConfig)
    assert isinstance(loaded["aligned"], CombinedStreamConfig)


def test_load_streams_rejects_duplicate_ids_across_roots(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    common_root = tmp_path / "common"
    _write_source_yaml(project_root / "sources", "data/*.jsonl")
    for root, configured_id in (
        (project_root / "streams", "prices"),
        (common_root / "streams", '" prices "'),
    ):
        root.mkdir(parents=True)
        (root / "prices.yaml").write_text(
            "\n".join(
                [
                    f"id: {configured_id}",
                    "from: {source: sample.fs}",
                    "map: {entrypoint: map_price}",
                ]
            ),
            encoding="utf-8",
        )
    project_yaml = _write_project_yaml(project_root)
    project_yaml.write_text(
        project_yaml.read_text(encoding="utf-8").replace(
            "  streams: streams",
            "  streams: [streams, ../common/streams]",
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Duplicate stream id 'prices'"):
        load_streams(load_project(project_yaml))


def test_load_streams_rejects_missing_id(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    _write_source_yaml(project_root / "sources", "data/*.jsonl")
    streams = project_root / "streams"
    streams.mkdir(parents=True)
    (streams / "bad.yaml").write_text(
        "from: {source: sample.fs}\nmap: {entrypoint: map_price}\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Missing 'id' in stream file"):
        load_streams(load_project(_write_project_yaml(project_root)))
