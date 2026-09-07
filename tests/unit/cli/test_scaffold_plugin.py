from pathlib import Path

import pytest
import yaml

from jerrythomas.io.yaml import read_yaml_document
from jerrythomas.profiles.loader import profile_specs_with_defaults
from jerrythomas.services.dataset import dataset_from_document
from jerrythomas.services.project import load_project
from jerrythomas.services.scaffold.plugin import scaffold_plugin
from jerrythomas.services.scaffold.templates import render
from jerrythomas.services.streams.loader import load_streams

_TEMPLATES_ROOT = Path(__file__).parents[3] / "src" / "jerrythomas" / "templates"
_PLUGIN_SKELETON_ROOT = _TEMPLATES_ROOT / "plugin_skeleton"
_DATASET_SKELETON_ROOT = _TEMPLATES_ROOT / "dataset_skeleton"


def test_template_sources_do_not_contain_generated_files() -> None:
    generated = sorted(
        path.relative_to(_TEMPLATES_ROOT)
        for path in _TEMPLATES_ROOT.rglob("*")
        if path.name in {"__pycache__", ".DS_Store"} or path.suffix in {".pyc", ".pyo"}
    )

    assert generated == []


def test_demo_package_overlay_does_not_replace_plugin_files() -> None:
    plugin_package = _PLUGIN_SKELETON_ROOT / "src" / "{{PACKAGE_NAME}}"
    demo_package = _TEMPLATES_ROOT / "demo_skeleton" / "src" / "demo"
    plugin_files = {
        path.relative_to(plugin_package)
        for path in plugin_package.rglob("*")
        if path.is_file()
    }
    demo_files = {
        path.relative_to(demo_package)
        for path in demo_package.rglob("*")
        if path.is_file()
    }

    assert plugin_files.isdisjoint(demo_files)


def test_loader_stub_renders_valid_python() -> None:
    template = "loaders/basic.py.j2"
    source = render(template, CLASS_NAME="ExampleLoader")

    compile(source, template, "exec")


def test_source_mapper_stub_renders_valid_explicit_python() -> None:
    source = render(
        "mappers/source.py.j2",
        FUNCTION_NAME="map_weather",
        INPUT_CLASS="WeatherDTO",
        INPUT_IMPORT="example.dtos",
        DOMAIN_MODULE="example.domains.weather",
        DOMAIN_RECORD="WeatherRecord",
    )

    compile(source, "mappers/source.py.j2", "exec")
    assert "**params" not in source
    assert "yield" not in source


def test_demo_stream_templates_use_v2_transform_fields() -> None:
    streams_root = _TEMPLATES_ROOT / "demo_skeleton" / "demo" / "streams"

    for path in streams_root.glob("*.yaml"):
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        assert "record" not in data, path
        assert "stream" not in data, path


def test_demo_stream_catalog_matches_config_models() -> None:
    project = _TEMPLATES_ROOT / "demo_skeleton" / "demo" / "project.yaml"

    config = load_streams(load_project(project))

    assert len(config.sources) == 2
    assert len(config.streams) == 5


@pytest.mark.parametrize(
    ("project", "cadence"),
    [
        (_DATASET_SKELETON_ROOT / "your-dataset" / "project.yaml", "1h"),
        (_TEMPLATES_ROOT / "demo_skeleton" / "demo" / "project.yaml", "1d"),
    ],
)
def test_template_dataset_cadence_resolves_from_project_global(
    project: Path,
    cadence: str,
) -> None:
    manifest = load_project(project)
    dataset = dataset_from_document(
        manifest,
        read_yaml_document(manifest.dataset_path),
    )

    assert dataset.sample.cadence == cadence


def test_demo_liquidity_filter_has_a_value_from_the_first_record() -> None:
    stream_path = (
        _TEMPLATES_ROOT / "demo_skeleton" / "demo" / "streams" / "equity.ohlcv.yaml"
    )
    transforms = yaml.safe_load(stream_path.read_text(encoding="utf-8"))["transforms"]

    assert transforms[0]["operation"] == "rolling"
    assert transforms[0]["to"] == "adv5"
    assert transforms[0]["min_samples"] == 1
    assert transforms[1]["operation"] == "where"
    assert transforms[1]["field"] == "adv5"


def test_plugin_template_has_one_minimal_dataset() -> None:
    assert (_DATASET_SKELETON_ROOT / "your-dataset").is_dir()
    assert not (_DATASET_SKELETON_ROOT / "your-interim-data-builder").exists()
    assert not (_DATASET_SKELETON_ROOT / "_dataset_base").exists()
    assert not (_DATASET_SKELETON_ROOT / "reference").exists()


def test_template_profiles_separate_builds_from_runtime_actions() -> None:
    demo_expected = {
        "build.defaults.yaml",
        "build.metadata.yaml",
        "build.scaler.yaml",
        "build.series.yaml",
        "inspect.coverage.yaml",
        "inspect.defaults.yaml",
        "inspect.matrix.yaml",
        "serve.defaults.yaml",
        "serve.dataset.yaml",
    }
    dataset_profiles = _DATASET_SKELETON_ROOT / "your-dataset" / "profiles"
    demo_profiles = _TEMPLATES_ROOT / "demo_skeleton" / "demo" / "profiles"

    assert {path.name for path in dataset_profiles.glob("*.yaml")} == {
        "inspect.coverage.yaml",
        "inspect.matrix.yaml",
        "serve.dataset.yaml",
        "serve.defaults.yaml",
    }
    assert {path.name for path in demo_profiles.glob("*.yaml")} == demo_expected

    project = load_project(_DATASET_SKELETON_ROOT / "your-dataset" / "project.yaml")
    serve_profiles, _ = profile_specs_with_defaults(project, cmd="serve")
    inspect_profiles, _ = profile_specs_with_defaults(project, cmd="inspect")
    assert [
        (profile.cmd, profile.name, profile.operation)
        for profile in [*serve_profiles, *inspect_profiles]
    ] == [
        ("serve", "dataset", "dataset"),
        ("inspect", "coverage", "coverage"),
        ("inspect", "matrix", "matrix"),
    ]


def test_scaffold_plugin_normalizes_hyphenated_name(tmp_path: Path) -> None:
    scaffold_plugin("test-datapipeline", tmp_path)

    plugin_root = tmp_path / "test-datapipeline"
    dataset_root = plugin_root / "your-dataset"
    assert (plugin_root / "src" / "test_datapipeline").is_dir()
    assert (dataset_root / ".env.example").is_file()
    assert not (plugin_root / "_dataset_base").exists()
    assert not (plugin_root / "reference").exists()
    assert not (plugin_root / "your-interim-data-builder").exists()
    assert not (dataset_root / "operations").exists()
    assert not (dataset_root / "profiles" / "build.schema.yaml").exists()
    assert (dataset_root / "profiles" / "inspect.coverage.yaml").is_file()
    assert (dataset_root / "profiles" / "inspect.matrix.yaml").is_file()
    assert (dataset_root / "profiles" / "serve.dataset.yaml").is_file()
    assert (dataset_root / "profiles" / "serve.defaults.yaml").is_file()
    assert (plugin_root / "src" / "test_datapipeline" / "combiners").is_dir()
    assert ".jerry-scaffold.lock" in (plugin_root / ".gitignore").read_text(
        encoding="utf-8"
    )

    pyproject = (plugin_root / "pyproject.toml").read_text()
    assert 'name = "test-datapipeline"' in pyproject
    assert '"jerry-thomas>=10.0.2"' in pyproject
    assert '[project.entry-points."jerrythomas.combiners"]' in pyproject

    readme = (plugin_root / "README.md").read_text()
    assert "python -m pip install -e ." in readme
    assert "{{" not in readme


def test_scaffold_plugin_normalizes_dotted_name(tmp_path: Path) -> None:
    scaffold_plugin("weather.data-plugin", tmp_path)

    assert (tmp_path / "weather.data-plugin" / "src" / "weather_data_plugin").is_dir()


def test_scaffold_plugin_creates_local_workspace(tmp_path: Path) -> None:
    scaffold_plugin("test-datapipeline", tmp_path)

    plugin_jerry = tmp_path / "test-datapipeline" / "jerry.yaml"

    assert plugin_jerry.is_file()
    assert not (tmp_path / "jerry.yaml").exists()

    cfg = yaml.safe_load(plugin_jerry.read_text())
    assert cfg["plugin_root"] == "."
    assert cfg["datasets"]["your-dataset"] == "your-dataset/project.yaml"
    assert set(cfg["datasets"]) == {"your-dataset"}


def test_scaffold_plugin_respects_outdir(tmp_path: Path) -> None:
    outdir = tmp_path / "plugins"
    outdir.mkdir()

    scaffold_plugin("myplugin", outdir)

    plugin_root = outdir / "myplugin"

    assert not (tmp_path / "jerry.yaml").exists()
    cfg = yaml.safe_load((plugin_root / "jerry.yaml").read_text())
    assert cfg["plugin_root"] == "."
    assert cfg["datasets"]["your-dataset"] == "your-dataset/project.yaml"
    assert set(cfg["datasets"]) == {"your-dataset"}


def test_scaffold_plugin_does_not_modify_parent_workspace(tmp_path: Path) -> None:
    workspace = tmp_path / "jerry.yaml"
    original = b"# keep formatting\ndatasets: {research: research/project.yaml}\n"
    workspace.write_bytes(original)

    scaffold_plugin("myplugin", tmp_path)

    assert workspace.read_bytes() == original
    assert (tmp_path / "myplugin" / "jerry.yaml").is_file()


def test_scaffold_plugin_does_not_remove_existing_target(tmp_path: Path) -> None:
    plugin_root = tmp_path / "myplugin"
    plugin_root.mkdir()
    sentinel = plugin_root / "keep.txt"
    sentinel.write_text("keep\n", encoding="utf-8")

    with pytest.raises(FileExistsError, match="already exists"):
        scaffold_plugin("myplugin", tmp_path)

    assert sentinel.read_text(encoding="utf-8") == "keep\n"


@pytest.mark.parametrize(
    "name",
    [
        "",
        "data pipeline",
        "jerrythomas",
        "JerryThomas",
        "jerry-thomas",
        "Jerry.Thomas",
        "jerry_thomas",
        "class",
        "café",
        "_plugin",
        "plugin_",
        ".plugin",
        "plugin.",
    ],
)
def test_scaffold_plugin_rejects_disallowed_names(name: str, tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        scaffold_plugin(name, tmp_path)
