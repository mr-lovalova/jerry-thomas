from types import SimpleNamespace

import pytest

from datapipeline.cli.command_router import execute_command
from datapipeline.cli.parser_builder import build_parser
from datapipeline.profiles.errors import ProfileCommandError


def _execute(args, workspace=None) -> None:
    execute_command(
        args=args,
        plugin_root=None,
        workspace_context=workspace,
        cli_level_arg="DEBUG",
        base_level_name="INFO",
        cli_log_outputs=[],
    )


def test_materialize_parser_accepts_profile_overrides() -> None:
    args = build_parser().parse_args(
        [
            "materialize",
            "--project",
            "project.yaml",
            "--profile",
            "adv-20",
            "--output",
            "adv-20.jsonl.gz",
            "--overwrite",
            "--artifact-mode",
            "force",
            "--visuals",
            "off",
            "--heartbeat-interval",
            "10",
        ]
    )

    assert args.cmd == "materialize"
    assert args.profile == "adv-20"
    assert args.output == "adv-20.jsonl.gz"
    assert args.overwrite is True
    assert args.artifact_mode == "FORCE"
    assert args.visuals == "off"
    assert args.heartbeat_interval_seconds == 10


def test_materialize_dispatches_one_profile_execution_path(monkeypatch) -> None:
    captured = {}
    monkeypatch.setattr(
        "datapipeline.cli.command_router.handle_materialize",
        lambda **kwargs: captured.update(kwargs),
    )
    args = build_parser().parse_args(
        [
            "materialize",
            "--project",
            "project.yaml",
            "--profile",
            "adv-20",
            "--output",
            "adv-20.jsonl",
            "--no-overwrite",
            "--artifact-mode",
            "auto",
        ]
    )

    _execute(args)
    assert captured == {
        "project": "project.yaml",
        "profile_name": "adv-20",
        "output": "adv-20.jsonl",
        "overwrite": False,
        "artifact_mode": "AUTO",
        "visuals": None,
        "heartbeat_interval_seconds": None,
        "cli_log_level": "DEBUG",
        "cli_log_outputs": [],
        "base_log_level": "INFO",
        "workspace": None,
    }


def test_materialize_output_override_requires_profile(monkeypatch) -> None:
    monkeypatch.setattr(
        "datapipeline.cli.commands.materialize.build_materialize_run_request",
        lambda **kwargs: pytest.fail("profiles should not run"),
    )
    args = build_parser().parse_args(
        ["materialize", "--project", "project.yaml", "--output", "out.jsonl"]
    )

    with pytest.raises(SystemExit) as exc_info:
        _execute(args)

    assert exc_info.value.code == 2


def test_materialize_resolves_profile_output_from_workspace(
    monkeypatch, tmp_path
) -> None:
    captured = {}
    request = object()
    monkeypatch.setattr(
        "datapipeline.cli.commands.materialize.build_materialize_run_request",
        lambda **kwargs: captured.update(kwargs) or request,
    )
    executed = []
    monkeypatch.setattr(
        "datapipeline.cli.commands.materialize.execute_profile_request",
        executed.append,
    )
    workspace = SimpleNamespace(root=tmp_path)
    args = build_parser().parse_args(
        [
            "materialize",
            "--project",
            "project.yaml",
            "--profile",
            "adv-20",
            "--output",
            "outputs/adv-20.jsonl",
            "--artifact-mode",
            "off",
        ]
    )

    _execute(args, workspace)
    assert captured["output"] == (tmp_path / "outputs/adv-20.jsonl").resolve()
    assert captured["profile_name"] == "adv-20"
    assert captured["artifact_mode"] == "OFF"
    assert executed == [request]


def test_materialize_passes_gzip_output_to_profile_resolution(monkeypatch) -> None:
    captured = {}
    request = object()
    monkeypatch.setattr(
        "datapipeline.cli.commands.materialize.build_materialize_run_request",
        lambda **kwargs: captured.update(kwargs) or request,
    )
    monkeypatch.setattr(
        "datapipeline.cli.commands.materialize.execute_profile_request",
        lambda selected: None,
    )
    args = build_parser().parse_args(
        [
            "materialize",
            "--project",
            "project.yaml",
            "--profile",
            "adv-20",
            "--output",
            "adv-20.jsonl.gz",
        ]
    )

    _execute(args)

    assert captured["output"].name == "adv-20.jsonl.gz"


def test_materialize_allows_global_overrides_without_profile(monkeypatch) -> None:
    captured = {}
    request = object()
    monkeypatch.setattr(
        "datapipeline.cli.commands.materialize.build_materialize_run_request",
        lambda **kwargs: captured.update(kwargs) or request,
    )
    monkeypatch.setattr(
        "datapipeline.cli.commands.materialize.execute_profile_request",
        lambda selected: None,
    )
    args = build_parser().parse_args(
        [
            "materialize",
            "--project",
            "project.yaml",
            "--overwrite",
            "--visuals",
            "off",
        ]
    )

    _execute(args)
    assert captured["profile_name"] is None
    assert captured["output"] is None
    assert captured["overwrite"] is True
    assert captured["cli_visuals"] == "off"


def test_materialize_profile_validation_error_reaches_cli_boundary(monkeypatch) -> None:
    def fail(**kwargs):
        raise ProfileCommandError("Unknown materialize profile 'missing'")

    monkeypatch.setattr(
        "datapipeline.cli.commands.materialize.build_materialize_run_request",
        fail,
    )
    args = build_parser().parse_args(
        [
            "materialize",
            "--project",
            "project.yaml",
            "--profile",
            "missing",
        ]
    )

    with pytest.raises(ProfileCommandError, match="Unknown materialize profile"):
        _execute(args)
