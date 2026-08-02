from pathlib import Path

from jerrythomas.cli.commands import demo, plugin
from jerrythomas.cli.workspace import WorkspaceContext
from jerrythomas.config.workspace import WorkspaceConfig


def _workspace_at(root: Path) -> WorkspaceContext:
    cfg = WorkspaceConfig.model_validate({})
    return WorkspaceContext(file_path=root / "jerry.yaml", config=cfg)


def test_plugin_init_out_path_is_workspace_relative(
    tmp_path: Path, monkeypatch
) -> None:
    workspace = _workspace_at(tmp_path)
    (tmp_path / "subdir").mkdir(parents=True, exist_ok=True)
    monkeypatch.chdir(tmp_path / "subdir")

    calls: dict[str, Path] = {}

    def fake_scaffold_plugin(name: str, outdir: Path) -> None:
        calls["name"] = name
        calls["outdir"] = outdir

    monkeypatch.setattr(plugin, "scaffold_plugin", fake_scaffold_plugin)

    plugin.handle("my-plugin", "plugins", workspace=workspace)

    assert calls["name"] == "my-plugin"
    assert calls["outdir"] == (tmp_path / "plugins").resolve()


def test_demo_init_out_path_is_workspace_relative(tmp_path: Path, monkeypatch) -> None:
    workspace = _workspace_at(tmp_path)
    (tmp_path / "nested" / "cwd").mkdir(parents=True, exist_ok=True)
    monkeypatch.chdir(tmp_path / "nested" / "cwd")

    calls: dict[str, Path] = {}

    def fake_scaffold_demo(outdir: Path) -> None:
        calls["demo_outdir"] = outdir

    monkeypatch.setattr(demo, "scaffold_demo", fake_scaffold_demo)

    demo.handle("create", out="plugins", workspace=workspace)

    expected_out = (tmp_path / "plugins").resolve()
    assert calls["demo_outdir"] == expected_out
