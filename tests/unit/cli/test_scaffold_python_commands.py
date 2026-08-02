from pathlib import Path

import pytest

from jerrythomas.cli.commands.mapper import handle as handle_mapper
from jerrythomas.cli.commands.parser import handle as handle_parser
from jerrythomas.plugins import MAPPERS_EP, PARSERS_EP


class PromptAborted(Exception):
    pass


def _raise_write_failure(*args, **kwargs):
    raise OSError("write failed")


def _create_plugin(tmp_path: Path) -> Path:
    root = tmp_path / "plugin"
    package = root / "src" / "sample_plugin"
    package.mkdir(parents=True)
    (package / "__init__.py").touch()
    (root / "pyproject.toml").write_text(
        '[project]\nname = "sample-plugin"\nversion = "0.1.0"\n',
        encoding="utf-8",
    )
    return root


def test_parser_finishes_prompting_before_creating_dto(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    plugin = _create_plugin(tmp_path)
    pyproject = plugin / "pyproject.toml"
    original_pyproject = pyproject.read_bytes()
    monkeypatch.setattr(
        "jerrythomas.cli.commands.parser.choose_dto",
        lambda existing, default=None: ("WeatherDTO", True),
    )

    def abort_parser_name(prompt: str, default: str | None = None) -> str:
        raise PromptAborted

    monkeypatch.setattr(
        "jerrythomas.cli.commands.parser.choose_name",
        abort_parser_name,
    )

    with pytest.raises(PromptAborted):
        handle_parser(None, plugin_root=plugin)

    package = plugin / "src" / "sample_plugin"
    assert not (package / "dtos").exists()
    assert not (package / "parsers").exists()
    assert pyproject.read_bytes() == original_pyproject


def test_mapper_finishes_prompting_before_creating_dependencies(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    plugin = _create_plugin(tmp_path)
    pyproject = plugin / "pyproject.toml"
    original_pyproject = pyproject.read_bytes()
    monkeypatch.setattr(
        "jerrythomas.cli.commands.mapper.pick_from_menu",
        lambda *args, **kwargs: "dto",
    )
    monkeypatch.setattr(
        "jerrythomas.cli.commands.mapper.choose_dto",
        lambda existing, default=None: ("WeatherDTO", True),
    )
    monkeypatch.setattr(
        "jerrythomas.cli.commands.mapper.choose_domain",
        lambda existing, default=None: ("weather", True),
    )

    def abort_mapper_name(prompt: str, default: str | None = None) -> str:
        raise PromptAborted

    monkeypatch.setattr(
        "jerrythomas.cli.commands.mapper.choose_name",
        abort_mapper_name,
    )

    with pytest.raises(PromptAborted):
        handle_mapper(None, plugin_root=plugin)

    package = plugin / "src" / "sample_plugin"
    assert not (package / "dtos").exists()
    assert not (package / "domains").exists()
    assert not (package / "mappers").exists()
    assert pyproject.read_bytes() == original_pyproject


def test_parser_validates_pyproject_before_creating_selected_dto(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    plugin = _create_plugin(tmp_path)
    pyproject = plugin / "pyproject.toml"
    pyproject.write_text("[project\n", encoding="utf-8")
    monkeypatch.setattr(
        "jerrythomas.cli.commands.parser.choose_dto",
        lambda existing, default=None: ("WeatherDTO", True),
    )

    with pytest.raises(SystemExit, match="Unexpected character"):
        handle_parser("WeatherParser", plugin_root=plugin)

    package = plugin / "src" / "sample_plugin"
    assert not (package / "dtos").exists()
    assert not (package / "parsers").exists()
    assert pyproject.read_text(encoding="utf-8") == "[project\n"


def test_mapper_validates_pyproject_before_creating_selected_dependencies(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    plugin = _create_plugin(tmp_path)
    pyproject = plugin / "pyproject.toml"
    pyproject.write_text("[project\n", encoding="utf-8")
    monkeypatch.setattr(
        "jerrythomas.cli.commands.mapper.pick_from_menu",
        lambda *args, **kwargs: "dto",
    )
    monkeypatch.setattr(
        "jerrythomas.cli.commands.mapper.choose_dto",
        lambda existing, default=None: ("WeatherDTO", True),
    )
    monkeypatch.setattr(
        "jerrythomas.cli.commands.mapper.choose_domain",
        lambda existing, default=None: ("weather", True),
    )

    with pytest.raises(SystemExit, match="Unexpected character"):
        handle_mapper("map_weather", plugin_root=plugin)

    package = plugin / "src" / "sample_plugin"
    assert not (package / "dtos").exists()
    assert not (package / "domains").exists()
    assert not (package / "mappers").exists()
    assert pyproject.read_text(encoding="utf-8") == "[project\n"


@pytest.mark.parametrize("collision", ["file", "entrypoint"])
def test_parser_collision_does_not_create_selected_dto(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    collision: str,
) -> None:
    plugin = _create_plugin(tmp_path)
    package = plugin / "src" / "sample_plugin"
    pyproject = plugin / "pyproject.toml"
    parser_path = package / "parsers" / "weather_parser.py"
    if collision == "file":
        parser_path.parent.mkdir()
        parser_path.write_text("existing parser\n", encoding="utf-8")
    else:
        with pyproject.open("a", encoding="utf-8") as project_file:
            project_file.write(
                f'\n[project.entry-points."{PARSERS_EP}"]\n'
                'weather_parser = "sample_plugin.parsers.weather_parser:'
                'WeatherParser"\n'
            )
    original_pyproject = pyproject.read_bytes()
    monkeypatch.setattr(
        "jerrythomas.cli.commands.parser.choose_dto",
        lambda existing, default=None: ("WeatherDTO", True),
    )

    with pytest.raises(SystemExit, match="already exists"):
        handle_parser("WeatherParser", plugin_root=plugin)

    assert not (package / "dtos").exists()
    if collision == "file":
        assert parser_path.read_text(encoding="utf-8") == "existing parser\n"
    assert pyproject.read_bytes() == original_pyproject


@pytest.mark.parametrize("collision", ["file", "entrypoint"])
def test_mapper_collision_does_not_create_selected_dependencies(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    collision: str,
) -> None:
    plugin = _create_plugin(tmp_path)
    package = plugin / "src" / "sample_plugin"
    pyproject = plugin / "pyproject.toml"
    mapper_path = package / "mappers" / "map_weather.py"
    if collision == "file":
        mapper_path.parent.mkdir()
        mapper_path.write_text("existing mapper\n", encoding="utf-8")
    else:
        with pyproject.open("a", encoding="utf-8") as project_file:
            project_file.write(
                f'\n[project.entry-points."{MAPPERS_EP}"]\n'
                'map_weather = "sample_plugin.mappers.map_weather:map_weather"\n'
            )
    original_pyproject = pyproject.read_bytes()
    monkeypatch.setattr(
        "jerrythomas.cli.commands.mapper.pick_from_menu",
        lambda *args, **kwargs: "dto",
    )
    monkeypatch.setattr(
        "jerrythomas.cli.commands.mapper.choose_dto",
        lambda existing, default=None: ("WeatherDTO", True),
    )
    monkeypatch.setattr(
        "jerrythomas.cli.commands.mapper.choose_domain",
        lambda existing, default=None: ("weather", True),
    )

    with pytest.raises(SystemExit, match="already exists"):
        handle_mapper("map_weather", plugin_root=plugin)

    assert not (package / "dtos").exists()
    assert not (package / "domains").exists()
    if collision == "file":
        assert mapper_path.read_text(encoding="utf-8") == "existing mapper\n"
    assert pyproject.read_bytes() == original_pyproject


def test_parser_rolls_back_created_dto_when_registration_fails(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    plugin = _create_plugin(tmp_path)
    package = plugin / "src" / "sample_plugin"
    pyproject = plugin / "pyproject.toml"
    original_pyproject = pyproject.read_bytes()
    monkeypatch.setattr(
        "jerrythomas.cli.commands.parser.choose_dto",
        lambda existing, default=None: ("WeatherDTO", True),
    )
    monkeypatch.setattr(
        "jerrythomas.services.scaffold.entrypoints._write_document",
        _raise_write_failure,
    )

    with pytest.raises(OSError, match="write failed"):
        handle_parser("WeatherParser", plugin_root=plugin)

    assert (package / "__init__.py").exists()
    assert not (package / "dtos").exists()
    assert not (package / "parsers").exists()
    assert pyproject.read_bytes() == original_pyproject


def test_mapper_rolls_back_created_dependencies_when_registration_fails(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    plugin = _create_plugin(tmp_path)
    package = plugin / "src" / "sample_plugin"
    pyproject = plugin / "pyproject.toml"
    original_pyproject = pyproject.read_bytes()
    monkeypatch.setattr(
        "jerrythomas.cli.commands.mapper.pick_from_menu",
        lambda *args, **kwargs: "dto",
    )
    monkeypatch.setattr(
        "jerrythomas.cli.commands.mapper.choose_dto",
        lambda existing, default=None: ("WeatherDTO", True),
    )
    monkeypatch.setattr(
        "jerrythomas.cli.commands.mapper.choose_domain",
        lambda existing, default=None: ("weather", True),
    )
    monkeypatch.setattr(
        "jerrythomas.services.scaffold.entrypoints._write_document",
        _raise_write_failure,
    )

    with pytest.raises(OSError, match="write failed"):
        handle_mapper("map_weather", plugin_root=plugin)

    assert (package / "__init__.py").exists()
    assert not (package / "dtos").exists()
    assert not (package / "domains").exists()
    assert not (package / "mappers").exists()
    assert pyproject.read_bytes() == original_pyproject
