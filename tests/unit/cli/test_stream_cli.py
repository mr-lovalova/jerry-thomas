from pathlib import Path

import pytest

from jerrythomas.cli.prompts import pick_from_menu, pick_multiple_from_list
from jerrythomas.cli.commands.inflow import (
    MapperSelection,
    ParserSelection,
    _collect_stream_plan,
    _mapper_menu_options,
    _parser_menu_options,
    _select_mapper_plan,
    _select_new_source_loader,
    _select_parser_plan,
)
from jerrythomas.services.scaffold.stream_plan import (
    DomainReference,
    MapperCreation,
    MapperReference,
    ParserCreation,
    ParserReference,
    PythonType,
    SourceCreation,
    SourceReference,
)


def test_parser_menu_options_include_existing_when_available() -> None:
    options = _parser_menu_options({"custom.parser": "custom.parser"})
    assert [key for key, _ in options] == [
        "create",
        "existing",
        "temporal_record",
        "identity",
    ]


def test_mapper_menu_options_skip_existing_when_empty() -> None:
    options = _mapper_menu_options({})
    assert [key for key, _ in options] == ["create", "identity"]


def test_pick_from_menu_blank_input_keeps_first_option_as_default(monkeypatch) -> None:
    monkeypatch.setattr("builtins.input", lambda _: "")

    choice = pick_from_menu(
        "Parser:",
        [
            ("identity", "Identity parser (default)"),
            ("temporal_record", "Temporal record rehydration"),
            ("custom", "Custom parser"),
        ],
    )

    assert choice == "identity"


def test_pick_multiple_rejects_partially_invalid_selection(monkeypatch) -> None:
    monkeypatch.setattr("builtins.input", lambda _: "1,999")

    with pytest.raises(SystemExit, match="Invalid selection"):
        pick_multiple_from_list("Streams:", ["first", "second"])


def test_select_new_source_loader_uses_builtin_transport(monkeypatch) -> None:
    choices = iter(["http", "json"])
    monkeypatch.setattr(
        "jerrythomas.cli.commands.inflow.pick_from_menu",
        lambda *args, **kwargs: next(choices),
    )

    loader = _select_new_source_loader()

    assert loader["transport"] == "http"
    reader = loader["reader"]
    assert isinstance(reader, dict)
    assert reader["format"] == "json"


def test_select_new_source_loader_accepts_custom_entrypoint(monkeypatch) -> None:
    monkeypatch.setattr(
        "jerrythomas.cli.commands.inflow.pick_from_menu",
        lambda *args, **kwargs: "custom",
    )
    monkeypatch.setattr(
        "jerrythomas.cli.commands.inflow.prompt_required",
        lambda prompt: "custom.loader",
    )

    loader = _select_new_source_loader()

    assert loader == {"entrypoint": "custom.loader", "args": {}}


def test_select_parser_plan_identity(monkeypatch) -> None:
    monkeypatch.setattr(
        "jerrythomas.cli.commands.inflow.list_parsers",
        lambda root=None: {},
    )
    monkeypatch.setattr(
        "jerrythomas.cli.commands.inflow.pick_from_menu",
        lambda *args, **kwargs: "identity",
    )

    selection = _select_parser_plan(
        Path("/tmp/plugin"), "example_pkg", "nasa", "weather"
    )

    assert selection.plan == ParserReference("identity")
    assert selection.dto_to_create is None


def test_select_parser_plan_temporal_record(monkeypatch) -> None:
    monkeypatch.setattr(
        "jerrythomas.cli.commands.inflow.list_parsers",
        lambda root=None: {},
    )
    monkeypatch.setattr(
        "jerrythomas.cli.commands.inflow.pick_from_menu",
        lambda *args, **kwargs: "temporal_record",
    )

    selection = _select_parser_plan(
        Path("/tmp/plugin"), "example_pkg", "nasa", "weather"
    )

    assert selection.plan == ParserReference("core.temporal_record")
    assert selection.dto_to_create is None


def test_select_parser_plan_uses_existing_entrypoint(monkeypatch) -> None:
    choices = iter(["existing", "custom.parser"])
    monkeypatch.setattr(
        "jerrythomas.cli.commands.inflow.list_parsers",
        lambda root=None: {"custom.parser": "custom.parser"},
    )
    monkeypatch.setattr(
        "jerrythomas.cli.commands.inflow.pick_from_menu",
        lambda *args, **kwargs: next(choices),
    )

    selection = _select_parser_plan(
        Path("/tmp/plugin"), "example_pkg", "nasa", "weather"
    )

    assert selection.plan == ParserReference("custom.parser")
    assert selection.dto_to_create is None


def test_select_parser_plan_creates_parser_and_dto(monkeypatch) -> None:
    monkeypatch.setattr(
        "jerrythomas.cli.commands.inflow.list_parsers",
        lambda root=None: {},
    )
    monkeypatch.setattr(
        "jerrythomas.cli.commands.inflow.pick_from_menu",
        lambda *args, **kwargs: "create",
    )
    monkeypatch.setattr(
        "jerrythomas.cli.commands.inflow.list_dtos",
        lambda root=None: {},
    )
    monkeypatch.setattr(
        "jerrythomas.cli.commands.inflow.choose_dto",
        lambda existing, default=None: ("WeatherDTO", True),
    )
    monkeypatch.setattr(
        "jerrythomas.cli.commands.inflow.choose_name",
        lambda *args, **kwargs: "WeatherDTOParser",
    )

    selection = _select_parser_plan(
        Path("/tmp/plugin"), "example_pkg", "nasa", "weather"
    )

    assert selection.plan == ParserCreation(
        "WeatherDTOParser",
        PythonType("WeatherDTO", "example_pkg.dtos.weather_dto"),
    )
    assert selection.dto_to_create == "WeatherDTO"


def test_existing_parser_dto_keeps_discovered_module(monkeypatch) -> None:
    monkeypatch.setattr(
        "jerrythomas.cli.commands.inflow.list_parsers",
        lambda root=None: {},
    )
    monkeypatch.setattr(
        "jerrythomas.cli.commands.inflow.pick_from_menu",
        lambda *args, **kwargs: "create",
    )
    monkeypatch.setattr(
        "jerrythomas.cli.commands.inflow.list_dtos",
        lambda root=None: {"WeatherDTO": "example_pkg.dtos.records"},
    )
    monkeypatch.setattr(
        "jerrythomas.cli.commands.inflow.choose_dto",
        lambda existing, default=None: ("WeatherDTO", False),
    )
    monkeypatch.setattr(
        "jerrythomas.cli.commands.inflow.choose_name",
        lambda *args, **kwargs: "WeatherDTOParser",
    )

    selection = _select_parser_plan(
        Path("/tmp/plugin"), "example_pkg", "nasa", "weather"
    )

    assert selection.plan == ParserCreation(
        "WeatherDTOParser",
        PythonType("WeatherDTO", "example_pkg.dtos.records"),
    )
    assert selection.dto_to_create is None


def test_existing_mapper_dto_keeps_discovered_module(monkeypatch) -> None:
    choices = iter(["create", "dto"])
    monkeypatch.setattr(
        "jerrythomas.cli.commands.inflow.list_mappers",
        lambda root=None: {},
    )
    monkeypatch.setattr(
        "jerrythomas.cli.commands.inflow.pick_from_menu",
        lambda *args, **kwargs: next(choices),
    )
    monkeypatch.setattr(
        "jerrythomas.cli.commands.inflow.list_dtos",
        lambda root=None: {"WeatherDTO": "example_pkg.dtos.records"},
    )
    monkeypatch.setattr(
        "jerrythomas.cli.commands.inflow.choose_dto",
        lambda existing, default=None: ("WeatherDTO", False),
    )
    monkeypatch.setattr(
        "jerrythomas.cli.commands.inflow.choose_name",
        lambda *args, **kwargs: "map_records_to_weather",
    )

    selection = _select_mapper_plan(
        Path("/tmp/plugin"),
        "example_pkg",
        "nasa",
        "weather",
        SourceReference("nasa.weather"),
        DomainReference("weather"),
    )

    assert selection.plan == MapperCreation(
        "map_records_to_weather",
        PythonType("WeatherDTO", "example_pkg.dtos.records"),
    )
    assert selection.dto_to_create is None


def test_mapper_reuses_new_parser_dto_without_second_creation(monkeypatch) -> None:
    choices = iter(["create", "dto"])
    monkeypatch.setattr(
        "jerrythomas.cli.commands.inflow.list_mappers",
        lambda root=None: {},
    )
    monkeypatch.setattr(
        "jerrythomas.cli.commands.inflow.pick_from_menu",
        lambda *args, **kwargs: next(choices),
    )

    def fail_dto_discovery(root=None):
        raise AssertionError("parser DTO must be reused")

    monkeypatch.setattr(
        "jerrythomas.cli.commands.inflow.list_dtos",
        fail_dto_discovery,
    )
    monkeypatch.setattr(
        "jerrythomas.cli.commands.inflow.choose_name",
        lambda *args, **kwargs: "map_weather_to_domain",
    )
    parser_dto = PythonType("WeatherDTO", "example_pkg.dtos.weather_dto")
    source = SourceCreation(
        source_id="nasa.weather",
        loader={"entrypoint": "custom.loader", "args": {}},
        parser=ParserCreation("WeatherParser", parser_dto),
    )

    selection = _select_mapper_plan(
        Path("/tmp/plugin"),
        "example_pkg",
        "nasa",
        "weather",
        source,
        DomainReference("weather"),
    )

    assert selection.plan == MapperCreation(
        "map_weather_to_domain",
        parser_dto,
    )
    assert selection.dto_to_create is None


def test_collect_stream_plan_preserves_custom_source_id(monkeypatch) -> None:
    prompts = iter(["nasa", "weather"])
    names = iter(["nasa.weather.hourly", "weather.hourly"])
    monkeypatch.setattr(
        "jerrythomas.cli.commands.inflow.prompt_required",
        lambda prompt: next(prompts),
    )
    monkeypatch.setattr(
        "jerrythomas.cli.commands.inflow.pick_from_menu",
        lambda *args, **kwargs: "create",
    )
    monkeypatch.setattr(
        "jerrythomas.cli.commands.inflow.choose_name",
        lambda *args, **kwargs: next(names),
    )
    monkeypatch.setattr(
        "jerrythomas.cli.commands.inflow._select_new_source_loader",
        lambda: ("custom.loader", {}),
    )
    parser_plan = ParserCreation(
        "WeatherParser",
        PythonType("WeatherDTO", "example_pkg.dtos.weather_dto"),
    )
    monkeypatch.setattr(
        "jerrythomas.cli.commands.inflow._select_parser_plan",
        lambda *args: ParserSelection(parser_plan, "WeatherDTO"),
    )
    monkeypatch.setattr(
        "jerrythomas.cli.commands.inflow.choose_domain",
        lambda existing, default=None: ("weather", False),
    )
    monkeypatch.setattr(
        "jerrythomas.cli.commands.inflow.list_domains",
        lambda root=None: ["weather"],
    )
    monkeypatch.setattr(
        "jerrythomas.cli.commands.inflow._select_mapper_plan",
        lambda *args: MapperSelection(MapperReference("identity"), None),
    )

    plan = _collect_stream_plan(
        Path("/tmp/plugin"),
        "example_pkg",
        Path("/tmp/project.yaml"),
    )

    assert isinstance(plan.source, SourceCreation)
    assert plan.source.source_id == "nasa.weather.hourly"
    assert plan.source.parser == parser_plan
    assert plan.mapper == MapperReference("identity")
    assert plan.dto_to_create == "WeatherDTO"
    assert plan.stream_id == "weather.hourly"


def test_collect_stream_plan_rejects_invalid_custom_source_before_loader(
    monkeypatch,
) -> None:
    prompts = iter(["nasa", "weather"])
    monkeypatch.setattr(
        "jerrythomas.cli.commands.inflow.prompt_required",
        lambda prompt: next(prompts),
    )
    monkeypatch.setattr(
        "jerrythomas.cli.commands.inflow.pick_from_menu",
        lambda *args, **kwargs: "create",
    )
    monkeypatch.setattr(
        "jerrythomas.cli.commands.inflow.choose_name",
        lambda label, default: "weather",
    )

    with pytest.raises(SystemExit, match="source_id"):
        _collect_stream_plan(
            Path("/tmp/plugin"),
            "example_pkg",
            Path("/tmp/project.yaml"),
        )
