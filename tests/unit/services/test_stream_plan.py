from pathlib import Path

import pytest

from datapipeline.plugins import PARSERS_EP
from datapipeline.services.scaffold.paths import ensure_project_scaffold
from datapipeline.services.scaffold.stream_plan import (
    DomainCreation,
    DomainReference,
    MapperCreation,
    MapperReference,
    ParserCreation,
    ParserReference,
    PythonType,
    SourceCreation,
    SourceReference,
    StreamPlan,
    execute_stream_plan,
)


def _patch_project(monkeypatch, tmp_path: Path) -> Path:
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text("[project]\nname = 'example'\n", encoding="utf-8")
    monkeypatch.setattr(
        "datapipeline.services.scaffold.stream_plan.pkg_root",
        lambda root: (tmp_path, "example", pyproject),
    )
    return pyproject


def test_custom_source_id_is_used_for_source_and_stream(monkeypatch, tmp_path) -> None:
    _patch_project(monkeypatch, tmp_path)
    created_source: dict[str, object] = {}
    created_stream: dict[str, object] = {}
    monkeypatch.setattr(
        "datapipeline.services.scaffold.stream_plan.create_source_yaml",
        lambda **kwargs: created_source.update(kwargs),
    )
    monkeypatch.setattr(
        "datapipeline.services.scaffold.stream_plan.write_source_stream",
        lambda **kwargs: created_stream.update(kwargs),
    )
    plan = StreamPlan(
        project_yaml=tmp_path / "project.yaml",
        stream_id="weather.hourly",
        root=tmp_path,
        source=SourceCreation(
            source_id="nasa.weather.hourly",
            loader={"entrypoint": "custom.loader", "args": {}},
            parser=ParserReference("identity"),
        ),
        mapper=MapperReference("identity"),
        domain=DomainReference("weather"),
        dto_to_create=None,
    )

    execute_stream_plan(plan)

    assert created_source["source_id"] == "nasa.weather.hourly"
    assert created_stream["source"] == "nasa.weather.hourly"


def test_invalid_source_id_fails_before_project_mutation(monkeypatch, tmp_path) -> None:
    def fail_project_resolution(root):
        raise AssertionError("invalid plans must fail before project resolution")

    monkeypatch.setattr(
        "datapipeline.services.scaffold.stream_plan.pkg_root",
        fail_project_resolution,
    )
    plan = StreamPlan(
        project_yaml=tmp_path / "project.yaml",
        stream_id="weather.weather",
        root=tmp_path,
        source=SourceCreation(
            source_id="weather",
            loader={"entrypoint": "custom.loader", "args": {}},
            parser=ParserReference("identity"),
        ),
        mapper=MapperReference("identity"),
        domain=DomainCreation("weather"),
        dto_to_create=None,
    )

    with pytest.raises(ValueError, match="source_id"):
        execute_stream_plan(plan)


def test_existing_project_with_missing_config_dirs_can_be_scaffolded(
    monkeypatch,
    tmp_path: Path,
) -> None:
    _patch_project(monkeypatch, tmp_path)
    project_yaml = tmp_path / "project.yaml"
    project_yaml.write_text(
        "schema_version: 4\n"
        "artifact_revision: 1\n"
        "name: default\n"
        "paths:\n"
        "  streams: ./streams\n"
        "  sources: ./sources\n"
        "  dataset: dataset.yaml\n"
        "  artifacts: ./artifacts\n"
        "  profiles: ./profiles\n"
        "globals: {}\n",
        encoding="utf-8",
    )
    plan = StreamPlan(
        project_yaml=project_yaml,
        stream_id="weather.weather",
        root=tmp_path,
        source=SourceCreation(
            source_id="demo.weather",
            loader={"entrypoint": "custom.loader", "args": {}},
            parser=ParserReference("identity"),
        ),
        mapper=MapperReference("identity"),
        domain=DomainReference("weather"),
        dto_to_create=None,
    )

    result = execute_stream_plan(plan)

    assert result.path == tmp_path / "streams" / "weather.weather.yaml"
    assert (tmp_path / "sources" / "demo.weather.yaml").exists()


def test_existing_source_fails_before_creating_planned_components(
    monkeypatch,
    tmp_path: Path,
) -> None:
    pyproject = _patch_project(monkeypatch, tmp_path)
    original_pyproject = pyproject.read_bytes()
    project_yaml = tmp_path / "project.yaml"
    project = ensure_project_scaffold(project_yaml)
    common_sources = tmp_path / "common" / "sources"
    common_sources.mkdir(parents=True)
    project_yaml.write_text(
        project_yaml.read_text(encoding="utf-8").replace(
            "  sources: ./sources",
            "  sources: [./sources, ./common/sources]",
        ),
        encoding="utf-8",
    )
    source_path = common_sources / "existing.yaml"
    source_path.write_text(
        "id: nasa.weather\n"
        "parser: {entrypoint: identity}\n"
        "loader: {entrypoint: identity}\n",
        encoding="utf-8",
    )
    dto = PythonType("WeatherDTO", "example.dtos.weather_dto")
    plan = StreamPlan(
        project_yaml=project_yaml,
        stream_id="weather.weather",
        root=tmp_path,
        source=SourceCreation(
            source_id="nasa.weather",
            loader={
                "transport": "http",
                "url": "https://example.test/data.jsonl",
                "reader": {"format": "jsonl"},
            },
            parser=ParserCreation("WeatherParser", dto),
        ),
        mapper=MapperCreation("map_weather", dto),
        domain=DomainCreation("weather"),
        dto_to_create="WeatherDTO",
    )

    with pytest.raises(FileExistsError, match="Source id 'nasa.weather'"):
        execute_stream_plan(plan)

    assert not (tmp_path / "src").exists()
    assert not (project.stream_dirs[0] / "weather.weather.yaml").exists()
    assert "id: nasa.weather" in source_path.read_text(encoding="utf-8")
    assert pyproject.read_bytes() == original_pyproject


def test_existing_stream_fails_before_creating_planned_components(
    monkeypatch,
    tmp_path: Path,
) -> None:
    pyproject = _patch_project(monkeypatch, tmp_path)
    original_pyproject = pyproject.read_bytes()
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
    stream_path = common_streams / "existing.yaml"
    stream_path.write_text(
        "id: weather.weather\n"
        "from: {source: nasa.weather}\n"
        "map: {entrypoint: identity}\n",
        encoding="utf-8",
    )
    dto = PythonType("WeatherDTO", "example.dtos.weather_dto")
    plan = StreamPlan(
        project_yaml=project_yaml,
        stream_id="weather.weather",
        root=tmp_path,
        source=SourceCreation(
            source_id="nasa.weather",
            loader={
                "transport": "http",
                "url": "https://example.test/data.jsonl",
                "reader": {"format": "jsonl"},
            },
            parser=ParserCreation("WeatherParser", dto),
        ),
        mapper=MapperCreation("map_weather", dto),
        domain=DomainCreation("weather"),
        dto_to_create="WeatherDTO",
    )

    with pytest.raises(FileExistsError, match="Stream id 'weather.weather'"):
        execute_stream_plan(plan)

    assert not (tmp_path / "src").exists()
    assert not (project.source_dirs[0] / "nasa.weather.yaml").exists()
    assert "id: weather.weather" in stream_path.read_text(encoding="utf-8")
    assert pyproject.read_bytes() == original_pyproject


def test_existing_parser_entrypoint_fails_before_stream_plan_mutation(
    monkeypatch,
    tmp_path: Path,
) -> None:
    pyproject = _patch_project(monkeypatch, tmp_path)
    with pyproject.open("a", encoding="utf-8") as project_file:
        project_file.write(
            f'\n[project.entry-points."{PARSERS_EP}"]\n'
            'weather_parser = "example.parsers.weather_parser:WeatherParser"\n'
        )
    original_pyproject = pyproject.read_bytes()
    project_yaml = tmp_path / "project.yaml"
    project = ensure_project_scaffold(project_yaml)
    dto = PythonType("WeatherDTO", "example.dtos.weather_dto")
    plan = StreamPlan(
        project_yaml=project_yaml,
        stream_id="weather.weather",
        root=tmp_path,
        source=SourceCreation(
            source_id="nasa.weather",
            loader={
                "transport": "http",
                "url": "https://example.test/data.jsonl",
                "reader": {"format": "jsonl"},
            },
            parser=ParserCreation("WeatherParser", dto),
        ),
        mapper=MapperCreation("map_weather", dto),
        domain=DomainCreation("weather"),
        dto_to_create="WeatherDTO",
    )

    with pytest.raises(FileExistsError, match="Parser entry point"):
        execute_stream_plan(plan)

    assert not (tmp_path / "src").exists()
    assert not (project.source_dirs[0] / "nasa.weather.yaml").exists()
    assert not (project.stream_dirs[0] / "weather.weather.yaml").exists()
    assert pyproject.read_bytes() == original_pyproject


def test_creation_plan_executes_exact_declared_components(
    monkeypatch, tmp_path
) -> None:
    _patch_project(monkeypatch, tmp_path)
    order: list[str] = []
    parser_call: dict[str, object] = {}
    source_call: dict[str, object] = {}
    mapper_call: dict[str, object] = {}
    stream_call: dict[str, object] = {}

    def create_domain(domain, root, **kwargs):
        order.append("domain")

    def create_dto(name, root, **kwargs):
        order.append("dto")

    def create_parser(**kwargs):
        order.append("parser")
        parser_call.update(kwargs)
        return "weather_parser"

    def create_source(**kwargs):
        order.append("source")
        source_call.update(kwargs)

    def create_mapper(**kwargs):
        order.append("mapper")
        mapper_call.update(kwargs)
        return "weather_mapper"

    def create_stream(**kwargs):
        order.append("stream")
        stream_call.update(kwargs)

    monkeypatch.setattr(
        "datapipeline.services.scaffold.stream_plan.create_domain",
        create_domain,
    )
    monkeypatch.setattr(
        "datapipeline.services.scaffold.stream_plan.create_dto",
        create_dto,
    )
    monkeypatch.setattr(
        "datapipeline.services.scaffold.stream_plan.create_parser",
        create_parser,
    )
    monkeypatch.setattr(
        "datapipeline.services.scaffold.stream_plan.create_source_yaml",
        create_source,
    )
    monkeypatch.setattr(
        "datapipeline.services.scaffold.stream_plan.create_mapper",
        create_mapper,
    )
    monkeypatch.setattr(
        "datapipeline.services.scaffold.stream_plan.write_source_stream",
        create_stream,
    )
    dto = PythonType("WeatherDTO", "example.dtos.weather_dto")
    plan = StreamPlan(
        project_yaml=tmp_path / "project.yaml",
        stream_id="weather.weather",
        root=tmp_path,
        source=SourceCreation(
            source_id="nasa.weather",
            loader={
                "transport": "http",
                "url": "https://example.test/data.jsonl",
                "reader": {"format": "jsonl"},
            },
            parser=ParserCreation("WeatherParser", dto),
        ),
        mapper=MapperCreation(
            "map_weather_to_domain",
            PythonType(dto.class_name, dto.module),
        ),
        domain=DomainCreation("weather"),
        dto_to_create="WeatherDTO",
    )

    execute_stream_plan(plan)

    assert order == [
        "domain",
        "dto",
        "parser",
        "mapper",
        "source",
        "stream",
    ]
    assert parser_call["dto_module"] == "example.dtos.weather_dto"
    assert source_call["parser_ep"] == "weather_parser"
    assert mapper_call["input_module"] == "example.dtos.weather_dto"
    assert stream_call["mapper_entrypoint"] == "weather_mapper"


def test_reference_plan_only_writes_stream(monkeypatch, tmp_path) -> None:
    _patch_project(monkeypatch, tmp_path)

    def fail_source_creation(**kwargs):
        raise AssertionError("source must be reused")

    monkeypatch.setattr(
        "datapipeline.services.scaffold.stream_plan.create_source_yaml",
        fail_source_creation,
    )
    captured: dict[str, object] = {}
    monkeypatch.setattr(
        "datapipeline.services.scaffold.stream_plan.write_source_stream",
        lambda **kwargs: captured.update(kwargs),
    )
    plan = StreamPlan(
        project_yaml=tmp_path / "project.yaml",
        stream_id="weather.weather",
        root=tmp_path,
        source=SourceReference("prices"),
        mapper=MapperReference("custom.mapper"),
        domain=DomainReference("weather"),
        dto_to_create=None,
    )

    execute_stream_plan(plan)

    assert captured["source"] == "prices"
    assert captured["mapper_entrypoint"] == "custom.mapper"


def test_dto_creation_is_a_single_explicit_plan_action(monkeypatch, tmp_path) -> None:
    _patch_project(monkeypatch, tmp_path)
    created_dto: dict[str, object] = {}
    mapper_call: dict[str, object] = {}

    def create_mapper(**kwargs):
        mapper_call.update(kwargs)
        return "weather_mapper"

    monkeypatch.setattr(
        "datapipeline.services.scaffold.stream_plan.create_dto",
        lambda name, root, **kwargs: created_dto.update(name=name, root=root),
    )
    monkeypatch.setattr(
        "datapipeline.services.scaffold.stream_plan.create_mapper",
        create_mapper,
    )
    monkeypatch.setattr(
        "datapipeline.services.scaffold.stream_plan.write_source_stream",
        lambda **kwargs: None,
    )
    plan = StreamPlan(
        project_yaml=tmp_path / "project.yaml",
        stream_id="weather.weather",
        root=tmp_path,
        source=SourceReference("nasa.weather"),
        mapper=MapperCreation(
            "map_weather_to_domain",
            PythonType("WeatherDTO", "example.dtos.weather_dto"),
        ),
        domain=DomainReference("weather"),
        dto_to_create="WeatherDTO",
    )

    execute_stream_plan(plan)

    assert created_dto["name"] == "WeatherDTO"
    assert mapper_call["input_module"] == "example.dtos.weather_dto"


def test_stream_plan_rolls_back_after_late_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    plugin = tmp_path / "plugin"
    package = plugin / "src" / "example"
    package.mkdir(parents=True)
    (package / "__init__.py").touch()
    pyproject = plugin / "pyproject.toml"
    pyproject.write_text("[project]\nname = 'example'\n", encoding="utf-8")
    original_pyproject = pyproject.read_bytes()
    project_yaml = plugin / "your-dataset" / "project.yaml"
    dto = PythonType("WeatherDTO", "example.dtos.weather_dto")
    plan = StreamPlan(
        project_yaml=project_yaml,
        stream_id="weather.hourly",
        root=plugin,
        source=SourceCreation(
            source_id="nasa.weather",
            loader={"entrypoint": "custom.loader", "args": {}},
            parser=ParserCreation("WeatherParser", dto),
        ),
        mapper=MapperCreation("map_weather", dto),
        domain=DomainCreation("weather"),
        dto_to_create="WeatherDTO",
    )

    def fail_stream_write(**kwargs):
        raise OSError("stream write failed")

    monkeypatch.setattr(
        "datapipeline.services.scaffold.stream_plan.write_source_stream",
        fail_stream_write,
    )

    with pytest.raises(OSError, match="stream write failed"):
        execute_stream_plan(plan)

    assert pyproject.read_bytes() == original_pyproject
    assert (package / "__init__.py").exists()
    assert not (package / "domains").exists()
    assert not (package / "dtos").exists()
    assert not (package / "parsers").exists()
    assert not (package / "mappers").exists()
    assert not project_yaml.exists()
    assert not (project_yaml.parent / ".env.example").exists()
    assert not (project_yaml.parent / "sources").exists()
    assert not (project_yaml.parent / "streams").exists()
    assert not (project_yaml.parent / "profiles").exists()
