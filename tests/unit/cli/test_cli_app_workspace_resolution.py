import sys
from pathlib import Path

import pytest
from datapipeline.cli import app
from datapipeline.cli.workspace import WorkspaceContext
from datapipeline.config.workspace import WorkspaceConfig
from datapipeline.profiles.errors import ProfileCommandError


def test_source_add_skips_dataset_resolution(monkeypatch, tmp_path):
    # Running source scaffolding should not attempt workspace dataset/project resolution.
    monkeypatch.chdir(tmp_path)

    def fail_resolution(*_args, **_kwargs):
        raise AssertionError(
            "project/dataset resolution should be skipped for source create"
        )

    monkeypatch.setattr(app, "_resolve_project_from_args", fail_resolution)

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
        "datapipeline.cli.commands.source.create_source_yaml", fake_create_source_yaml
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
    assert captured["parser_ep"] == "identity"


def test_dataset_path_resolves_relative_to_workspace_root(monkeypatch, tmp_path):
    workspace_root = tmp_path
    nested = workspace_root / "nested" / "cwd"
    nested.mkdir(parents=True)
    project_file = workspace_root / "projects" / "weather" / "project.yaml"
    project_file.parent.mkdir(parents=True)
    project_file.write_text(
        "schema_version: 4\nartifact_revision: 1\nname: weather\npaths: {}\n",
        encoding="utf-8",
    )

    workspace = WorkspaceContext(
        file_path=workspace_root / "jerry.yaml",
        config=WorkspaceConfig.model_validate({}),
    )
    monkeypatch.chdir(nested)

    resolved = app._dataset_to_project_path("projects/weather/project.yaml", workspace)
    assert Path(resolved) == project_file.resolve()


def test_resolve_project_from_args_rejects_project_and_dataset():
    try:
        app._resolve_project_from_args("project.yaml", "alias", None)
    except SystemExit as exc:
        assert "Cannot use both --project and --dataset" in str(exc)
    else:
        raise AssertionError(
            "Expected SystemExit when both project and dataset are set"
        )


def test_resolve_project_from_args_uses_workspace_default_dataset(tmp_path):
    project_file = tmp_path / "datasets" / "demo" / "project.yaml"
    project_file.parent.mkdir(parents=True)
    project_file.write_text(
        "schema_version: 4\nartifact_revision: 1\nname: demo\npaths: {}\n",
        encoding="utf-8",
    )

    workspace = WorkspaceContext(
        file_path=tmp_path / "jerry.yaml",
        config=WorkspaceConfig.model_validate(
            {
                "datasets": {"demo": "datasets/demo/project.yaml"},
                "default_dataset": "demo",
            }
        ),
    )

    project = app._resolve_project_from_args(None, None, workspace)
    assert Path(project) == project_file.resolve()


def test_resolve_project_from_args_prefers_explicit_project_over_workspace_default(
    tmp_path,
):
    workspace = WorkspaceContext(
        file_path=tmp_path / "jerry.yaml",
        config=WorkspaceConfig.model_validate(
            {
                "datasets": {"demo": "datasets/demo/project.yaml"},
                "default_dataset": "demo",
            }
        ),
    )

    project = app._resolve_project_from_args("custom/project.yaml", None, workspace)
    assert project == "custom/project.yaml"


def test_resolve_project_from_args_requires_selection_without_workspace_default():
    workspace = WorkspaceContext(
        file_path=Path("/tmp/jerry.yaml"),
        config=WorkspaceConfig.model_validate({}),
    )
    try:
        app._resolve_project_from_args(None, None, workspace)
    except SystemExit as exc:
        assert "No dataset/project selected" in str(exc)
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
        "schema_version: 4\nartifact_revision: 1\nname: demo\npaths: {}\n",
        encoding="utf-8",
    )
    workspace = WorkspaceContext(
        file_path=tmp_path / "jerry.yaml",
        config=WorkspaceConfig.model_validate(
            {
                "datasets": {"demo": "datasets/demo/project.yaml"},
                "default_dataset": "demo",
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
