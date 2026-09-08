from pathlib import Path
from types import SimpleNamespace

import pytest

from jerrythomas.cli.command_router import execute_command
from jerrythomas.cli.commands import list_ as list_command
from jerrythomas.cli.parser_builder import build_parser
from jerrythomas.cli.workspace import WorkspaceContext
from jerrythomas.config.preview import PREVIEW_STAGES
from jerrythomas.config.workspace import WorkspaceConfig
from jerrythomas.profiles.errors import ProfileCommandError


def _execute(args, *, plugin_root=None, workspace=None) -> None:
    execute_command(
        args=args,
        plugin_root=plugin_root,
        workspace_context=workspace,
        cli_level_arg=None,
        cli_log_outputs=[],
    )


def test_plugin_name_dispatches_from_positional_argument(monkeypatch) -> None:
    captured: dict[str, object] = {}
    monkeypatch.setattr(
        "jerrythomas.cli.command_router.handle_plugin",
        lambda **kwargs: captured.update(kwargs),
    )

    _execute(build_parser().parse_args(["plugin", "create", "weather-plugin"]))

    assert captured["name"] == "weather-plugin"


def test_domain_name_dispatches_from_positional_argument(monkeypatch) -> None:
    captured: dict[str, object] = {}
    monkeypatch.setattr(
        "jerrythomas.cli.command_router.handle_domain",
        lambda **kwargs: captured.update(kwargs),
    )

    _execute(build_parser().parse_args(["domain", "create", "weather"]))

    assert captured["domain"] == "weather"


@pytest.mark.parametrize(
    ("argv", "subcmd"),
    [
        (["list", "domains"], "domains"),
        (["list", "sources"], "sources"),
    ],
)
def test_list_route_forwards_plugin_and_workspace(
    monkeypatch,
    tmp_path,
    argv,
    subcmd,
) -> None:
    plugin_root = tmp_path / "plugin"
    workspace = WorkspaceContext(
        file_path=tmp_path / "jerry.yaml",
        config=WorkspaceConfig.model_validate({}),
    )
    captured: dict[str, object] = {}
    monkeypatch.setattr(
        "jerrythomas.cli.command_router.handle_list",
        lambda **kwargs: captured.update(kwargs),
    )

    _execute(
        build_parser().parse_args(argv),
        plugin_root=plugin_root,
        workspace=workspace,
    )

    assert captured == {
        "subcmd": subcmd,
        "plugin_root": plugin_root,
        "workspace": workspace,
    }


@pytest.mark.parametrize(
    "argv",
    [
        [
            "--log-level",
            "DEBUG",
            "--log-output",
            "stdout",
            "serve",
            "--project",
            "project.yaml",
        ],
        [
            "serve",
            "--log-level",
            "DEBUG",
            "--log-output",
            "stdout",
            "--heartbeat-interval",
            "5",
            "--project",
            "project.yaml",
        ],
    ],
)
def test_common_options_survive_before_or_after_command(argv) -> None:
    args = build_parser().parse_args(argv)

    assert args.log_level == "DEBUG"
    assert args.log_output == ["stdout"]


@pytest.mark.parametrize("command", ["serve", "build", "inspect", "materialize"])
def test_execution_commands_accept_observability_flags(command) -> None:
    args = build_parser().parse_args(
        [command, "--no-visuals", "--heartbeat-interval", "5"]
    )

    assert args.visuals is False
    assert args.heartbeat_interval_seconds == 5


def test_profile_request_builder_error_reaches_cli_boundary_without_lifecycle(
    monkeypatch,
) -> None:
    executed: list[object] = []
    error = ProfileCommandError("invalid serve profile")

    def fail(**_kwargs):
        raise error

    monkeypatch.setattr(
        "jerrythomas.cli.commands.profile_runner.build_runtime_run_request",
        fail,
    )
    monkeypatch.setattr(
        "jerrythomas.cli.commands.profile_runner.execute_profile_request",
        executed.append,
    )
    with pytest.raises(ProfileCommandError) as raised:
        _execute(
            build_parser().parse_args(
                ["serve", "--project", "project.yaml"],
            )
        )

    assert raised.value is error
    assert executed == []


def test_heartbeat_is_an_execution_command_option() -> None:
    with pytest.raises(SystemExit) as exc:
        build_parser().parse_args(["--heartbeat-interval", "5", "serve"])

    assert exc.value.code == 2


@pytest.mark.parametrize(
    "argv",
    [
        ["clean"],
        ["demo", "create"],
        ["list", "domains"],
        ["list", "sources"],
    ],
)
def test_non_execution_commands_reject_heartbeat(argv) -> None:
    with pytest.raises(SystemExit) as exc:
        build_parser().parse_args([*argv, "--heartbeat-interval", "5"])

    assert exc.value.code == 2


@pytest.mark.parametrize("value", ["-1", "nan", "inf", "-inf"])
def test_heartbeat_rejects_invalid_values(value) -> None:
    with pytest.raises(SystemExit) as exc:
        build_parser().parse_args(["serve", "--heartbeat-interval", value])

    assert exc.value.code == 2


@pytest.mark.parametrize("command", ["serve", "inspect"])
@pytest.mark.parametrize("value", ["0", "-1", "1.5", "many"])
def test_runtime_limit_must_be_a_positive_integer(command, value) -> None:
    with pytest.raises(SystemExit) as exc:
        build_parser().parse_args([command, "--limit", value])

    assert exc.value.code == 2


def test_runtime_limit_accepts_a_positive_integer() -> None:
    args = build_parser().parse_args(["serve", "--limit", "1"])

    assert args.limit == 1


@pytest.mark.parametrize("command", ["serve", "build", "inspect", "materialize"])
def test_profile_flag_selects_a_profile(command) -> None:
    args = build_parser().parse_args([command, "--profile", "disabled-profile"])

    assert args.profile == "disabled-profile"


def test_profile_help_explains_explicit_disabled_selection(capsys) -> None:
    with pytest.raises(SystemExit) as exc:
        build_parser().parse_args(["serve", "--help"])

    assert exc.value.code == 0
    help_text = " ".join(capsys.readouterr().out.split())
    assert "explicitly selected disabled profiles still run" in help_text


@pytest.mark.parametrize("preview", PREVIEW_STAGES)
def test_serve_parser_accepts_semantic_preview_stages(preview) -> None:
    args = build_parser().parse_args(["serve", "--preview", preview])

    assert args.preview == preview


def test_serve_parser_rejects_numeric_preview(capsys) -> None:
    with pytest.raises(SystemExit) as exc:
        build_parser().parse_args(["serve", "--preview", "3"])

    assert exc.value.code == 2


def test_source_listing_uses_workspace_project_without_python_package(
    monkeypatch,
    tmp_path,
    capsys,
) -> None:
    project_yaml = tmp_path / "dataset" / "project.yaml"
    monkeypatch.setattr(
        list_command,
        "resolve_default_project_yaml",
        lambda workspace: project_yaml,
    )
    monkeypatch.setattr(
        list_command,
        "load_project",
        lambda path: path,
    )
    monkeypatch.setattr(
        list_command,
        "load_streams",
        lambda project: SimpleNamespace(sources={"nasa.weather": object()}),
    )
    monkeypatch.setattr(
        list_command,
        "pkg_root",
        lambda root: (_ for _ in ()).throw(
            AssertionError("workspace source listing must not require a Python package")
        ),
    )

    list_command.handle("sources", workspace=object())

    assert capsys.readouterr().out.strip() == "nasa.weather"


@pytest.mark.parametrize(
    ("subcmd", "function_name"),
    [
        ("domains", "list_domains"),
        ("dtos", "list_dtos"),
        ("parsers", "list_parsers"),
        ("mappers", "list_mappers"),
        ("combiners", "list_combiners"),
        ("loaders", "list_loaders"),
    ],
)
def test_plugin_discovery_lists_use_configured_root(
    monkeypatch,
    tmp_path,
    subcmd,
    function_name,
) -> None:
    plugin_root = tmp_path / "plugin"
    seen: list[Path | None] = []
    monkeypatch.setattr(
        list_command,
        function_name,
        lambda *, root: seen.append(root) or {},
    )

    list_command.handle(subcmd, plugin_root=plugin_root)

    assert seen == [plugin_root]


@pytest.mark.parametrize("command", ["serve", "inspect", "materialize"])
@pytest.mark.parametrize("value", ["AUTO", "FORCE", "OFF", "force", "off"])
def test_artifact_mode_cli_rejects_legacy_values(command, value) -> None:
    with pytest.raises(SystemExit) as exc:
        build_parser().parse_args([command, "--artifact-mode", value])
    assert exc.value.code == 2


@pytest.mark.parametrize("command", ["serve", "inspect", "materialize", "build"])
@pytest.mark.parametrize("value", ["on", "off"])
def test_visuals_cli_rejects_legacy_value_arguments(command, value) -> None:
    with pytest.raises(SystemExit) as exc:
        build_parser().parse_args([command, "--visuals", value])
    assert exc.value.code == 2
