import pytest

from jerrythomas.artifacts.series import load_series_manifest
from jerrythomas.cli.command_router import execute_command
from jerrythomas.cli.parser_builder import build_parser
from jerrythomas.profiles.errors import ProfileCommandError


def test_build_cli_artifact_modes_control_real_cached_outputs(copy_fixture):
    root = copy_fixture("regression_project")
    (root / "profiles" / "build.series.yaml").write_text(
        "operation: dataset.default.series\n", encoding="utf-8"
    )
    manifest_path = root / "build" / "datasets" / "default" / "series" / "manifest.json"

    def build(mode):
        args = build_parser().parse_args(
            [
                "build",
                "--project",
                str(root / "project.yaml"),
                "--profile",
                "series",
                "--artifact-mode",
                mode,
                "--no-visuals",
            ]
        )
        execute_command(
            args=args,
            plugin_root=None,
            workspace_context=None,
            cli_level_arg="CRITICAL",
            cli_log_outputs=[],
        )

    with pytest.raises(SystemExit) as failure:
        build("require_current")
    assert failure.value.code == 2
    assert isinstance(failure.value.__cause__, ProfileCommandError)
    assert "missing or stale" in str(failure.value.__cause__)
    assert not manifest_path.exists()

    build("auto")
    original = manifest_path.read_bytes()
    original_mtime = manifest_path.stat().st_mtime_ns
    original_generation = load_series_manifest(manifest_path).path
    for mode in ("auto", "require_current"):
        build(mode)
        assert manifest_path.read_bytes() == original
        assert manifest_path.stat().st_mtime_ns == original_mtime

    build("rebuild")
    assert load_series_manifest(manifest_path).path != original_generation
