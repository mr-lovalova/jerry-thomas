from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from jerrythomas.config.streams import (
    AlignedStreamConfig,
    BroadcastStreamConfig,
    SourceStreamConfig,
)
from jerrythomas.services.scaffold.paths import ensure_project_scaffold
from jerrythomas.services.scaffold.stream_yaml import (
    write_aligned_stream,
    write_broadcast_stream,
    write_source_stream,
)


@pytest.mark.parametrize(
    "stream_id",
    ["../outside", "/absolute", "nested/name", "a..b"],
)
def test_source_stream_rejects_unsafe_id_before_mutating_project(
    tmp_path: Path,
    stream_id: str,
) -> None:
    project_yaml = tmp_path / "project.yaml"

    with pytest.raises(ValidationError):
        write_source_stream(
            project_yaml,
            stream_id,
            "provider.prices",
            "identity",
        )

    assert not project_yaml.exists()
    assert not (tmp_path / "streams").exists()


def test_aligned_stream_rejects_invalid_inputs_before_mutating_project(
    tmp_path: Path,
) -> None:
    project_yaml = tmp_path / "project.yaml"

    with pytest.raises(ValidationError):
        write_aligned_stream(
            project_yaml,
            "prices.spread",
            ["prices.bid", "prices.bid"],
            "combine_spread",
        )

    assert not project_yaml.exists()
    assert not (tmp_path / "streams").exists()


def test_source_stream_refuses_to_replace_existing_config(tmp_path: Path) -> None:
    project_yaml = tmp_path / "project.yaml"
    ensure_project_scaffold(project_yaml)
    path = tmp_path / "streams" / "prices.daily.yaml"
    path.write_text("existing\n", encoding="utf-8")

    with pytest.raises(FileExistsError):
        write_source_stream(
            project_yaml,
            "prices.daily",
            "provider.prices",
            "identity",
        )

    assert path.read_text(encoding="utf-8") == "existing\n"


def test_aligned_stream_refuses_to_replace_existing_config(tmp_path: Path) -> None:
    project_yaml = tmp_path / "project.yaml"
    ensure_project_scaffold(project_yaml)
    path = tmp_path / "streams" / "prices.spread.yaml"
    path.write_text("existing\n", encoding="utf-8")

    with pytest.raises(FileExistsError):
        write_aligned_stream(
            project_yaml,
            "prices.spread",
            ["prices.bid", "prices.ask"],
            "combine_spread",
        )

    assert path.read_text(encoding="utf-8") == "existing\n"


def test_stream_scaffold_rejects_existing_id_in_another_root(tmp_path: Path) -> None:
    project_yaml = tmp_path / "project.yaml"
    project = ensure_project_scaffold(project_yaml)
    common_streams = tmp_path / "common" / "streams"
    common_streams.mkdir(parents=True)
    project_yaml.write_text(
        project_yaml.read_text(encoding="utf-8").replace(
            "  streams: ./streams",
            "  streams: [./streams, ./common/streams]",
        ),
        encoding="utf-8",
    )
    (common_streams / "existing.yaml").write_text(
        "id: prices.daily\n"
        "from: {source: provider.prices}\n"
        "map: {entrypoint: identity}\n",
        encoding="utf-8",
    )

    with pytest.raises(FileExistsError, match="Stream id 'prices.daily'"):
        write_source_stream(
            project_yaml,
            "prices.daily",
            "provider.prices",
            "identity",
        )

    assert not (project.stream_dirs[0] / "prices.daily.yaml").exists()


def test_stream_scaffold_does_not_validate_unrelated_stream_values(
    tmp_path: Path,
) -> None:
    project_yaml = tmp_path / "project.yaml"
    project = ensure_project_scaffold(project_yaml)
    (project_yaml.parent / ".env").write_text(
        "SECRET=TOP-SECRET-VALUE\n",
        encoding="utf-8",
    )
    (project.stream_dirs[0] / "existing.yaml").write_text(
        "id: prices.existing\n"
        "from: {source: '${env:SECRET}'}\n"
        "map: {entrypoint: identity}\n",
        encoding="utf-8",
    )

    path = write_source_stream(
        project_yaml,
        "prices.daily",
        "provider.prices",
        "identity",
    )

    assert path.name == "prices.daily.yaml"


def test_stream_scaffolds_render_valid_configs(tmp_path: Path) -> None:
    project_yaml = tmp_path / "project.yaml"

    source_path = write_source_stream(
        project_yaml,
        "prices.daily-us",
        "provider.prices",
        "identity",
    )
    aligned_path = write_aligned_stream(
        project_yaml,
        "prices.spread",
        ["prices.bid", "prices.ask"],
        "combine_spread",
    )
    broadcast_path = write_broadcast_stream(
        project_yaml,
        "prices.adjusted",
        "prices.daily",
        "market.adjustment",
        "apply_adjustment",
    )

    source = SourceStreamConfig.model_validate(
        yaml.safe_load(source_path.read_text(encoding="utf-8"))
    )
    aligned = AlignedStreamConfig.model_validate(
        yaml.safe_load(aligned_path.read_text(encoding="utf-8"))
    )
    broadcast = BroadcastStreamConfig.model_validate(
        yaml.safe_load(broadcast_path.read_text(encoding="utf-8"))
    )
    assert source.id == "prices.daily-us"
    assert aligned.from_.align == ("prices.bid", "prices.ask")
    assert broadcast.from_.stream == "prices.daily"
    assert broadcast.from_.broadcast == "market.adjustment"
    assert broadcast.combine.entrypoint == "apply_adjustment"


def test_stream_scaffolds_quote_yaml_sensitive_strings(tmp_path: Path) -> None:
    project_yaml = tmp_path / "project.yaml"

    source_path = write_source_stream(
        project_yaml,
        "null",
        "null",
        "null",
    )
    aligned_path = write_aligned_stream(
        project_yaml,
        "yes",
        ["null", "off"],
        "null",
    )
    broadcast_path = write_broadcast_stream(
        project_yaml,
        "on",
        "null",
        "off",
        "yes",
    )

    source = SourceStreamConfig.model_validate(
        yaml.safe_load(source_path.read_text(encoding="utf-8"))
    )
    aligned = AlignedStreamConfig.model_validate(
        yaml.safe_load(aligned_path.read_text(encoding="utf-8"))
    )
    broadcast = BroadcastStreamConfig.model_validate(
        yaml.safe_load(broadcast_path.read_text(encoding="utf-8"))
    )

    assert source.id == "null"
    assert source.from_.source == "null"
    assert source.map.entrypoint == "null"
    assert aligned.id == "yes"
    assert aligned.from_.align == ("null", "off")
    assert aligned.combine.entrypoint == "null"
    assert broadcast.id == "on"
    assert broadcast.from_.stream == "null"
    assert broadcast.from_.broadcast == "off"
    assert broadcast.combine.entrypoint == "yes"
