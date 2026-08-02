from pathlib import Path

import pytest
import yaml

from jerrythomas.config.workspace import WorkspaceConfig
from jerrythomas.plugins import MAPPERS_EP, PARSERS_EP
from jerrythomas.services.project import load_project
from jerrythomas.services.scaffold import demo
from jerrythomas.services.scaffold.entrypoints import read_entry_points
from jerrythomas.services.streams.loader import load_streams


def test_demo_scaffold_creates_self_contained_plugin(tmp_path: Path) -> None:
    plugin_root = demo.scaffold_demo(tmp_path)

    assert plugin_root == tmp_path / "demo"
    assert not (plugin_root / "your-dataset").exists()
    assert (plugin_root / "demo" / "project.yaml").is_file()
    assert (plugin_root / "demo" / "scripts" / "run_model_batches.py").is_file()
    assert (plugin_root / "src" / "demo" / "domains" / "equity").is_dir()
    assert "your-dataset" not in (plugin_root / "README.md").read_text()

    workspace = WorkspaceConfig.model_validate(
        yaml.safe_load((plugin_root / "jerry.yaml").read_text(encoding="utf-8"))
    )
    assert workspace.plugin_root == "."
    assert workspace.datasets == {"demo": "demo/project.yaml"}
    assert workspace.default_dataset == "demo"
    assert not (tmp_path / "jerry.yaml").exists()

    pyproject = plugin_root / "pyproject.toml"
    assert read_entry_points(pyproject, PARSERS_EP) == {
        "sandbox_ohlcv_dto_parser": (
            "demo.parsers.sandbox_ohlcv_dto_parser:SandboxOhlcvDTOParser"
        )
    }
    assert read_entry_points(pyproject, MAPPERS_EP) == {
        "map_sandbox_ohlcv_dto_to_equity": (
            "demo.mappers.map_sandbox_ohlcv_dto_to_equity:"
            "map_sandbox_ohlcv_dto_to_equity"
        )
    }

    project = load_project(plugin_root / "demo" / "project.yaml")
    streams = load_streams(project)
    assert len(streams.sources) == 2
    assert len(streams.streams) == 5

    generated_text = "\n".join(
        path.read_text(encoding="utf-8")
        for path in plugin_root.rglob("*")
        if path.is_file() and path.suffix in {".md", ".py", ".toml", ".yaml", ".yml"}
    )
    assert "{{" not in generated_text

    template_root = (
        Path(__file__).parents[3]
        / "src"
        / "jerrythomas"
        / "templates"
        / "demo_skeleton"
    )
    expected = {
        Path("README.md"),
        Path("jerry.yaml"),
        *(
            Path("demo") / path.relative_to(template_root / "demo")
            for path in (template_root / "demo").rglob("*")
            if path.is_file()
        ),
        *(
            Path("src/demo") / path.relative_to(template_root / "src" / "demo")
            for path in (template_root / "src" / "demo").rglob("*")
            if path.is_file()
        ),
    }
    assert all((plugin_root / path).is_file() for path in expected)


def test_demo_scaffold_refuses_existing_target_without_mutation(
    tmp_path: Path,
) -> None:
    plugin_root = tmp_path / "demo"
    plugin_root.mkdir()
    sentinel = plugin_root / "keep.txt"
    sentinel.write_text("keep\n", encoding="utf-8")

    with pytest.raises(FileExistsError, match="already exists"):
        demo.scaffold_demo(tmp_path)

    assert sentinel.read_text(encoding="utf-8") == "keep\n"


def test_demo_scaffold_does_not_modify_parent_workspace(tmp_path: Path) -> None:
    workspace_path = tmp_path / "jerry.yaml"
    original = b"# keep this comment\ndatasets: {research: research/project.yaml}\n"
    workspace_path.write_bytes(original)

    plugin_root = demo.scaffold_demo(tmp_path)

    assert workspace_path.read_bytes() == original
    assert (plugin_root / "jerry.yaml").is_file()


def test_demo_scaffold_removes_new_target_when_installation_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_install(plugin_root: Path, template_root: Path) -> None:
        raise RuntimeError("broken demo template")

    monkeypatch.setattr(demo, "_install_demo_files", fail_install)

    with pytest.raises(RuntimeError, match="broken demo template"):
        demo.scaffold_demo(tmp_path)

    assert not (tmp_path / "demo").exists()
