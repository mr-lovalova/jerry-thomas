from pathlib import Path

import pytest
from tomlkit.exceptions import ParseError

from jerrythomas.plugins import LOADERS_EP
from jerrythomas.services.scaffold.entrypoints import read_entry_points
from jerrythomas.services.scaffold.loader import create_loader
from jerrythomas.services.scaffold.mapper import create_mapper
from jerrythomas.services.scaffold.parser import create_parser


def _plugin_with_invalid_pyproject(tmp_path: Path) -> Path:
    plugin = tmp_path / "plugin"
    package = plugin / "src" / "plugin"
    package.mkdir(parents=True)
    (package / "__init__.py").touch()
    (plugin / "pyproject.toml").write_text("[project\n", encoding="utf-8")
    return plugin


def _valid_plugin(tmp_path: Path) -> Path:
    plugin = tmp_path / "plugin"
    package = plugin / "src" / "plugin"
    package.mkdir(parents=True)
    (package / "__init__.py").touch()
    (plugin / "pyproject.toml").write_text(
        "[project]\nname = 'plugin'\n",
        encoding="utf-8",
    )
    return plugin


def test_loader_validates_pyproject_before_creating_files(tmp_path: Path) -> None:
    plugin = _plugin_with_invalid_pyproject(tmp_path)

    with pytest.raises(ParseError):
        create_loader(name="WeatherLoader", root=plugin)

    assert not (plugin / "src" / "plugin" / "loaders").exists()


def test_parser_validates_pyproject_before_creating_files(tmp_path: Path) -> None:
    plugin = _plugin_with_invalid_pyproject(tmp_path)

    with pytest.raises(ParseError):
        create_parser(
            name="WeatherParser",
            dto_class="WeatherDTO",
            dto_module="plugin.dtos.weather",
            root=plugin,
        )

    assert not (plugin / "src" / "plugin" / "parsers").exists()


def test_mapper_validates_pyproject_before_creating_files(tmp_path: Path) -> None:
    plugin = _plugin_with_invalid_pyproject(tmp_path)

    with pytest.raises(ParseError):
        create_mapper(
            name="map_weather",
            input_class="WeatherDTO",
            input_module="plugin.dtos.weather",
            domain="weather",
            root=plugin,
        )

    assert not (plugin / "src" / "plugin" / "mappers").exists()


def test_loader_rejects_python_keyword_name(tmp_path: Path) -> None:
    plugin = tmp_path / "plugin"
    package = plugin / "src" / "plugin"
    package.mkdir(parents=True)
    (package / "__init__.py").touch()
    (plugin / "pyproject.toml").write_text(
        "[project]\nname = 'plugin'\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="valid Python identifier"):
        create_loader(name="class", root=plugin)

    assert not (package / "loaders").exists()


@pytest.mark.parametrize(
    ("name", "module_name", "class_name"),
    [
        ("weather", "weather", "WeatherLoader"),
        ("custom_loader", "custom_loader", "CustomLoader"),
        ("WeatherLoader", "weather_loader", "WeatherLoader"),
        ("_1", "_1", "_1Loader"),
    ],
)
def test_loader_generates_canonical_class_and_entry_point(
    tmp_path: Path,
    name: str,
    module_name: str,
    class_name: str,
) -> None:
    plugin = _valid_plugin(tmp_path)

    entry_point = create_loader(name=name, root=plugin)

    source_path = plugin / "src" / "plugin" / "loaders" / f"{module_name}.py"
    source = source_path.read_text(encoding="utf-8")
    compile(source, str(source_path), "exec")
    assert f"class {class_name}(BaseDataLoader):" in source
    assert read_entry_points(plugin / "pyproject.toml", LOADERS_EP) == {
        entry_point: f"plugin.loaders.{module_name}:{class_name}"
    }


def test_loader_removes_created_files_when_registration_fails(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    plugin = _valid_plugin(tmp_path)
    pyproject = plugin / "pyproject.toml"
    original = pyproject.read_bytes()

    def fail_write(*args, **kwargs):
        raise OSError("write failed")

    monkeypatch.setattr(
        "jerrythomas.services.scaffold.entrypoints._write_document",
        fail_write,
    )

    with pytest.raises(OSError, match="write failed"):
        create_loader(name="WeatherLoader", root=plugin)

    assert not (plugin / "src" / "plugin" / "loaders").exists()
    assert pyproject.read_bytes() == original
