import sys
from pathlib import Path

import pytest
from jerrythomas.cli import app
from jerrythomas.cli.parser_builder import build_parser
from jerrythomas.cli.workspace import WorkspaceContext
from jerrythomas.config.workspace import WorkspaceConfig
from jerrythomas.profiles.errors import ProfileCommandError


def _write_build_project(tmp_path: Path, enabled: bool = True) -> Path:
    project = tmp_path / "project.yaml"
    project.write_text(
        (
            "schema_version: 7\n"
            "artifact_revision: 1\n"
            "paths:\n"
            "  streams: streams\n"
            "  sources: sources\n"
            "  datasets: datasets\n"
            "  artifacts: artifacts\n"
            "  profiles: profiles\n"
        ),
        encoding="utf-8",
    )
    (tmp_path / "datasets").mkdir()
    (tmp_path / "datasets" / "default.yaml").write_text(
        "version: v1\nsample: {rounding: ceil, cadence: 1h}\n",
        encoding="utf-8",
    )
    for directory in ("profiles", "sources", "streams"):
        (tmp_path / directory).mkdir()
    (tmp_path / "profiles" / "build.metadata.yaml").write_text(
        f"operation: dataset.default.metadata\nenabled: {str(enabled).lower()}\n",
        encoding="utf-8",
    )
    return project


def test_source_create_without_project_keeps_plugin_fallback(monkeypatch, tmp_path):
    # Omitted project selection retains plugin scaffold fallback.
    monkeypatch.chdir(tmp_path)

    def fail_resolution(*_args, **_kwargs):
        raise AssertionError(
            "required project resolution should be skipped for source create"
        )

    monkeypatch.setattr(app, "resolve_project_from_args", fail_resolution)

    captured = {}

    def fake_create_source_yaml(
        source_id,
        loader,
        parser_ep,
        parser_args=None,
        root=None,
        project_yaml=None,
    ):
        captured.update(
            {
                "source_id": source_id,
                "loader": loader,
                "parser_ep": parser_ep,
                "root": root,
            }
        )

    monkeypatch.setattr(
        "jerrythomas.cli.commands.source.create_source_yaml", fake_create_source_yaml
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "jerry",
            "source",
            "create",
            "stooq.ohlcv_daily",
            "-t",
            "http",
            "-f",
            "json",
        ],
    )

    app.main()

    assert captured["source_id"] == "stooq.ohlcv_daily"
    assert captured["loader"]["transport"] == "http"
    assert captured["loader"]["reader"]["format"] == "json"
    assert captured["root"] is None
    assert captured["parser_ep"] == "core.identity"


def test_project_path_resolves_relative_to_workspace_root(monkeypatch, tmp_path):
    workspace_root = tmp_path
    nested = workspace_root / "nested" / "cwd"
    nested.mkdir(parents=True)
    project_file = workspace_root / "projects" / "weather" / "project.yaml"
    project_file.parent.mkdir(parents=True)
    project_file.write_text(
        "schema_version: 7\nartifact_revision: 1\nname: weather\npaths: {}\n",
        encoding="utf-8",
    )

    workspace = WorkspaceContext(
        file_path=workspace_root / "jerry.yaml",
        config=WorkspaceConfig.model_validate({}),
    )
    monkeypatch.chdir(nested)

    resolved = app.resolve_project_from_args("projects/weather/project.yaml", workspace)
    assert Path(resolved) == project_file.resolve()


@pytest.mark.parametrize("command", ["serve", "build", "inspect", "materialize"])
@pytest.mark.parametrize("flag", ["--dataset", "-d"])
def test_commands_reject_removed_dataset_selector(command, flag):
    with pytest.raises(SystemExit) as error:
        build_parser().parse_args([command, flag, "default"])
    assert error.value.code == 2


def test_resolve_project_from_args_uses_workspace_default_project(tmp_path):
    project_file = tmp_path / "datasets" / "demo" / "project.yaml"
    project_file.parent.mkdir(parents=True)
    project_file.write_text(
        "schema_version: 7\nartifact_revision: 1\nname: demo\npaths: {}\n",
        encoding="utf-8",
    )

    workspace = WorkspaceContext(
        file_path=tmp_path / "jerry.yaml",
        config=WorkspaceConfig.model_validate(
            {
                "projects": {"demo": "datasets/demo/project.yaml"},
                "default_project": "demo",
            }
        ),
    )

    project = app.resolve_project_from_args(None, workspace)
    assert Path(project) == project_file.resolve()


def test_resolve_project_from_args_prefers_explicit_project_over_workspace_default(
    tmp_path,
):
    workspace = WorkspaceContext(
        file_path=tmp_path / "jerry.yaml",
        config=WorkspaceConfig.model_validate(
            {
                "projects": {"demo": "datasets/demo/project.yaml"},
                "default_project": "demo",
            }
        ),
    )

    project = app.resolve_project_from_args("custom/project.yaml", workspace)
    assert project == str(tmp_path / "custom/project.yaml")


def test_resolve_project_from_args_requires_selection_without_workspace_default():
    workspace = WorkspaceContext(
        file_path=Path("/tmp/jerry.yaml"),
        config=WorkspaceConfig.model_validate({}),
    )
    try:
        app.resolve_project_from_args(None, workspace)
    except SystemExit as exc:
        assert "No project selected" in str(exc)
    else:
        raise AssertionError(
            "Expected SystemExit when no project selection is available"
        )


def test_main_handles_keyboard_interrupt_at_top_level(monkeypatch, capsys):
    monkeypatch.setattr(app, "load_workspace_context", lambda _cwd: None)
    monkeypatch.setattr(
        app,
        "execute_command",
        lambda **_kwargs: (_ for _ in ()).throw(KeyboardInterrupt()),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["jerry", "serve", "--project", "project.yaml"],
    )

    with pytest.raises(SystemExit) as exc:
        app.main()

    captured = capsys.readouterr()
    assert exc.value.code == 130
    assert "Serve interrupted by user" in captured.err


def test_main_converts_profile_error_to_cli_exit(monkeypatch):
    error = ProfileCommandError("invalid serve profile")
    error.add_note("additional profile detail")
    messages: list[str] = []

    def fail(**_kwargs) -> None:
        raise error

    monkeypatch.setattr(app, "load_workspace_context", lambda _cwd: None)
    monkeypatch.setattr(app, "execute_command", fail)
    monkeypatch.setattr(
        app.logger,
        "error",
        lambda message, *args: messages.append(message % args),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["jerry", "serve", "--project", "project.yaml"],
    )

    with pytest.raises(SystemExit) as raised:
        app.main()

    assert raised.value.code == 2
    assert raised.value.__cause__ is error
    assert messages == ["invalid serve profile", "additional profile detail"]


def test_invalid_cli_log_target_does_not_modify_artifact(
    monkeypatch,
    tmp_path,
) -> None:
    project = _write_build_project(tmp_path)
    artifact = tmp_path / "artifacts" / "build" / "metadata.json"
    artifact.parent.mkdir(parents=True)
    original = b'{"valid": true}\n'
    artifact.write_bytes(original)

    monkeypatch.setattr(app, "load_workspace_context", lambda _cwd: None)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "jerry",
            "--log-output",
            f"fs:{artifact}",
            "build",
            "--project",
            str(project),
        ],
    )

    with pytest.raises(SystemExit) as raised:
        app.main()

    assert raised.value.code == 2
    assert artifact.read_bytes() == original


def test_noop_profile_command_does_not_activate_cli_file_logging(
    monkeypatch,
    tmp_path,
    capsys,
) -> None:
    project = _write_build_project(tmp_path, enabled=False)
    log_path = tmp_path / "logs" / "build.log"
    monkeypatch.setattr(app, "load_workspace_context", lambda _cwd: None)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "jerry",
            "--log-output",
            f"fs:{log_path}",
            "build",
            "--project",
            str(project),
        ],
    )

    app.main()

    assert not log_path.exists()
    assert "No enabled build profiles; skipping build." in capsys.readouterr().err


def test_noop_profile_command_preserves_cli_stdout_logging(
    monkeypatch,
    tmp_path,
    capsys,
) -> None:
    project = _write_build_project(tmp_path, enabled=False)
    log_path = tmp_path / "logs" / "build.log"
    monkeypatch.setattr(app, "load_workspace_context", lambda _cwd: None)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "jerry",
            "--log-output",
            "stdout",
            "--log-output",
            f"fs:{log_path}",
            "build",
            "--project",
            str(project),
        ],
    )

    app.main()

    captured = capsys.readouterr()
    assert "No enabled build profiles; skipping build." in captured.out
    assert captured.err == ""
    assert not log_path.exists()


def test_noop_profile_command_does_not_modify_artifact_through_logging(
    monkeypatch,
    tmp_path,
) -> None:
    project = _write_build_project(tmp_path, enabled=False)
    artifact = tmp_path / "artifacts" / "build" / "metadata.json"
    artifact.parent.mkdir(parents=True)
    original = b'{"valid": true}\n'
    artifact.write_bytes(original)

    monkeypatch.setattr(app, "load_workspace_context", lambda _cwd: None)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "jerry",
            "--log-output",
            f"fs:{artifact}",
            "build",
            "--project",
            str(project),
        ],
    )

    app.main()

    assert artifact.read_bytes() == original


def test_main_parses_help_before_loading_workspace(monkeypatch, capsys):
    def fail_workspace_load(_cwd):
        raise AssertionError("help must not load the workspace")

    monkeypatch.setattr(app, "load_workspace_context", fail_workspace_load)
    monkeypatch.setattr(sys, "argv", ["jerry", "--help"])

    with pytest.raises(SystemExit) as exc:
        app.main()

    assert exc.value.code == 0
    assert "usage: jerry" in capsys.readouterr().out


@pytest.mark.parametrize("command", ["version", "env", "clean"])
def test_workspace_independent_commands_do_not_load_workspace(
    monkeypatch,
    command,
):
    def fail_workspace_load(_cwd):
        raise AssertionError(f"{command} must not load the workspace")

    monkeypatch.setattr(app, "load_workspace_context", fail_workspace_load)
    monkeypatch.setattr(app, "execute_command", lambda **_kwargs: True)
    monkeypatch.setattr(sys, "argv", ["jerry", command])

    app.main()


def test_main_reports_invalid_workspace_as_cli_error(monkeypatch, capsys):
    def load_invalid_workspace(_cwd):
        raise ValueError("invalid jerry.yaml")

    monkeypatch.setattr(app, "load_workspace_context", load_invalid_workspace)
    monkeypatch.setattr(sys, "argv", ["jerry", "list", "sources"])

    with pytest.raises(SystemExit) as exc:
        app.main()

    assert exc.value.code == 2
    assert "Failed to load workspace: invalid jerry.yaml" in capsys.readouterr().err


def test_main_resolves_project_for_serve_with_workspace_default(monkeypatch, tmp_path):
    project_file = tmp_path / "datasets" / "demo" / "project.yaml"
    project_file.parent.mkdir(parents=True)
    project_file.write_text(
        "schema_version: 7\nartifact_revision: 1\nname: demo\npaths: {}\n",
        encoding="utf-8",
    )
    workspace = WorkspaceContext(
        file_path=tmp_path / "jerry.yaml",
        config=WorkspaceConfig.model_validate(
            {
                "projects": {"demo": "datasets/demo/project.yaml"},
                "default_project": "demo",
            }
        ),
    )
    monkeypatch.setattr(app, "load_workspace_context", lambda _cwd: workspace)
    captured: dict[str, object] = {}

    def _capture_execute_command(**kwargs):
        captured["project"] = kwargs["args"].project
        return True

    monkeypatch.setattr(app, "execute_command", _capture_execute_command)
    monkeypatch.setattr(sys, "argv", ["jerry", "serve"])

    app.main()

    assert Path(captured["project"]) == project_file.resolve()


@pytest.mark.parametrize(
    "selector", ["research", "projects/research", "projects/research/project.yaml"]
)
def test_project_selector_accepts_alias_folder_or_yaml(tmp_path, selector):
    project = tmp_path / "projects" / "research" / "project.yaml"
    project.parent.mkdir(parents=True)
    project.touch()
    workspace = WorkspaceContext(
        file_path=tmp_path / "jerry.yaml",
        config=WorkspaceConfig(projects={"research": "projects/research"}),
    )

    assert app.resolve_project_from_args(selector, workspace) == str(project)


@pytest.mark.parametrize("resource", ["datasets", "streams", "profiles"])
def test_list_catalog_commands_accept_project_selector(resource):
    args = build_parser().parse_args(["list", resource, "--project", "research"])
    assert args.project == "research"
