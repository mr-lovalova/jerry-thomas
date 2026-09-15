import logging
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest

from jerrythomas.cli.command_router import execute_command
from jerrythomas.cli.commands.profile_runner import execute_profile_request
from jerrythomas.cli.output_options import build_cli_output_config
from jerrythomas.cli.parser_builder import build_parser
from jerrythomas.config.execution import ExecutionConfig
from jerrythomas.config.profiles.output import (
    ServeOutputConfig,
    merge_output_overrides,
)
from jerrythomas.execution.observability import CommandFinished
from jerrythomas.execution.settings import CommandObservability
from jerrythomas.profiles.errors import ProfileCommandError
from jerrythomas.profiles.models import BuildRunRequest


def _serve_args() -> SimpleNamespace:
    return SimpleNamespace(
        cmd="serve",
        result_json=False,
        project="project.yaml",
        limit=None,
        profile=None,
        preview=None,
        output_transport=None,
        output_format=None,
        output_directory=None,
        output_encoding=None,
        output_compression=None,
        output_view=None,
        artifact_mode=None,
        visuals=True,
        heartbeat_interval_seconds=None,
    )


def _inspect_args() -> SimpleNamespace:
    return SimpleNamespace(
        cmd="inspect",
        project="project.yaml",
        profile=None,
        limit=None,
        output_transport=None,
        output_format=None,
        output_directory=None,
        output_encoding=None,
        output_compression=None,
        output_view=None,
        artifact_mode=None,
        visuals=True,
        heartbeat_interval_seconds=None,
    )


def _runtime_request(visuals: bool = False) -> SimpleNamespace:
    return SimpleNamespace(
        command="serve",
        artifact_settings=SimpleNamespace(
            observability=SimpleNamespace(visuals=visuals)
        ),
        jobs=(),
    )


def _build_run_request(visuals: bool = False) -> BuildRunRequest:
    job = SimpleNamespace(
        settings=SimpleNamespace(
            observability=SimpleNamespace(visuals=visuals),
        )
    )
    return BuildRunRequest(
        definition=SimpleNamespace(),
        jobs=(job,),
        execution=ExecutionConfig(),
    )


@contextmanager
def _noop_visual_summary(_level, _enabled):
    yield


def test_build_cli_output_config_returns_none_without_flags() -> None:
    assert build_cli_output_config(None, None, None) is None


def test_build_cli_output_config_collects_provided_fs_leaves() -> None:
    overrides = build_cli_output_config("FS", "jsonl", "artifacts")

    assert overrides == {
        "transport": "fs",
        "format": "jsonl",
        "directory": Path("artifacts").resolve(),
    }


def test_build_cli_output_config_collects_partial_leaves() -> None:
    assert build_cli_output_config(None, None, "out") == {
        "directory": Path("out").resolve()
    }
    assert build_cli_output_config(None, None, None, output_compression="gzip") == {
        "compression": "gzip"
    }
    assert build_cli_output_config(None, None, None, view="flat") == {"view": "flat"}
    assert build_cli_output_config(
        "stdout", None, None, output_encoding="utf-8-sig"
    ) == {"transport": "stdout", "encoding": "utf-8-sig"}


def test_merge_output_overrides_directory_only_inherits_profile_leaves() -> None:
    base = ServeOutputConfig.model_validate(
        {"transport": "fs", "format": "jsonl", "directory": "artifacts/serve"}
    )

    merged = merge_output_overrides(base, {"directory": Path("elsewhere")})

    assert merged is not None
    assert merged.transport == "fs"
    assert merged.format == "jsonl"
    assert merged.encoding is None
    assert merged.directory == Path("elsewhere")


def test_merge_output_overrides_gzip_alone_applies_to_jsonl_profile() -> None:
    base = ServeOutputConfig.model_validate(
        {"transport": "fs", "format": "jsonl", "directory": "out"}
    )

    merged = merge_output_overrides(base, {"compression": "gzip"})

    assert merged is not None
    assert merged.compression == "gzip"


@pytest.mark.parametrize("format_", ["parquet", "pickle", "html"])
def test_format_override_does_not_inherit_default_text_encoding(format_) -> None:
    base = ServeOutputConfig(transport="fs", format="jsonl", directory=Path("out"))

    merged = merge_output_overrides(base, {"format": format_})

    assert merged is not None
    assert merged.format == format_
    assert merged.encoding is None
    assert merged.directory == Path("out")


@pytest.mark.parametrize("format_", ["parquet", "pickle", "html"])
def test_format_override_still_rejects_explicit_text_encoding(format_) -> None:
    base = ServeOutputConfig(
        transport="fs", format="jsonl", directory=Path("out"), encoding="utf-8"
    )

    with pytest.raises(ValueError, match=f"{format_} output does not support encoding"):
        merge_output_overrides(base, {"format": format_})


def test_merge_output_overrides_rejects_invalid_combinations() -> None:
    with pytest.raises(ValueError, match="fs outputs require a directory"):
        merge_output_overrides(None, {"transport": "fs", "format": "jsonl"})

    stdout_jsonl = ServeOutputConfig.model_validate(
        {"transport": "stdout", "format": "jsonl"}
    )
    with pytest.raises(ValueError, match="stdout cannot define a directory"):
        merge_output_overrides(stdout_jsonl, {"directory": "out"})

    fs_jsonl = ServeOutputConfig.model_validate(
        {"transport": "fs", "format": "jsonl", "directory": "out"}
    )
    with pytest.raises(ValueError, match="supports only jsonl and csv"):
        merge_output_overrides(fs_jsonl, {"format": "parquet", "compression": "gzip"})
    with pytest.raises(ValueError, match="csv output supports only view"):
        merge_output_overrides(fs_jsonl, {"format": "csv", "view": "raw"})
    with pytest.raises(ValueError, match="pickle output supports only view"):
        merge_output_overrides(fs_jsonl, {"format": "pickle", "view": "flat"})
    with pytest.raises(ValueError, match="stdout outputs do not support encoding"):
        merge_output_overrides(stdout_jsonl, {"encoding": "utf-8"})
    with pytest.raises(ValueError, match="encoding must name a registered codec"):
        merge_output_overrides(fs_jsonl, {"encoding": "definitely-not-a-codec"})
    with pytest.raises(ValueError, match="invalid output configuration"):
        merge_output_overrides(fs_jsonl, {"transport": "s3"})


def test_execute_serve_propagates_keyboard_interrupt(monkeypatch) -> None:
    monkeypatch.setattr(
        "jerrythomas.cli.commands.profile_runner.build_runtime_run_request",
        lambda **kwargs: object(),
    )

    calls = {"execute_profile_request": 0}

    def _interrupting_execute(request):
        calls["execute_profile_request"] += 1
        raise KeyboardInterrupt()

    monkeypatch.setattr(
        "jerrythomas.cli.commands.profile_runner.execute_profile_request",
        _interrupting_execute,
    )

    with pytest.raises(KeyboardInterrupt):
        execute_command(
            args=_serve_args(),
            plugin_root=None,
            workspace_context=None,
            cli_level_arg=None,
            cli_log_outputs=[],
        )

    assert calls["execute_profile_request"] == 1


def test_execute_serve_runs_request_from_builder(monkeypatch) -> None:
    sentinel_request = object()
    captured: dict[str, object] = {}

    def _build_request(**kwargs):
        captured.update(kwargs)
        return sentinel_request

    monkeypatch.setattr(
        "jerrythomas.cli.commands.profile_runner.build_runtime_run_request",
        _build_request,
    )

    seen = {"request": None}

    def _capture(request):
        seen["request"] = request

    monkeypatch.setattr(
        "jerrythomas.cli.commands.profile_runner.execute_profile_request",
        _capture,
    )

    args = _serve_args()
    args.artifact_mode = "rebuild"
    result = execute_command(
        args=args,
        plugin_root=None,
        workspace_context=None,
        cli_level_arg=None,
        cli_log_outputs=[],
    )

    assert result is None
    assert seen["request"] is sentinel_request
    assert captured["command"] == "serve"
    assert captured["artifact_mode"] == "rebuild"
    assert captured["command_observability"] == CommandObservability(visuals=True)


@pytest.mark.parametrize("command", ["serve", "inspect"])
def test_runtime_command_propagates_gzip_output_override(
    monkeypatch,
    tmp_path,
    command,
) -> None:
    captured: dict[str, object] = {}

    def _capture_request(**kwargs):
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(
        "jerrythomas.cli.commands.profile_runner.build_runtime_run_request",
        _capture_request,
    )
    monkeypatch.setattr(
        "jerrythomas.cli.commands.profile_runner.execute_profile_request",
        lambda request: None,
    )
    args = build_parser().parse_args(
        [
            command,
            "--output-transport",
            "fs",
            "--output-format",
            "jsonl",
            "--output-directory",
            str(tmp_path),
            "--output-compression",
            "gzip",
        ]
    )

    execute_command(
        args=args,
        plugin_root=None,
        workspace_context=None,
        cli_level_arg=None,
        cli_log_outputs=[],
    )

    output = captured["cli_output"]
    assert output == {
        "transport": "fs",
        "format": "jsonl",
        "directory": tmp_path,
        "compression": "gzip",
    }


def test_execute_serve_skips_when_no_enabled_profiles(monkeypatch, caplog) -> None:
    monkeypatch.setattr(
        "jerrythomas.cli.commands.profile_runner.build_runtime_run_request",
        lambda **kwargs: None,
    )

    with caplog.at_level(
        logging.INFO, logger="jerrythomas.cli.commands.profile_runner"
    ):
        result = execute_command(
            args=_serve_args(),
            plugin_root=None,
            workspace_context=None,
            cli_level_arg=None,
            cli_log_outputs=[],
        )

    assert result is None
    assert "No enabled serve profiles; skipping serve." in caplog.text


def test_execute_build_passes_profile_and_force(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def _capture_request(**kwargs):
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(
        "jerrythomas.cli.commands.profile_runner.build_build_run_request",
        _capture_request,
    )
    monkeypatch.setattr(
        "jerrythomas.cli.commands.profile_runner.execute_profile_request",
        lambda request: None,
    )

    args = SimpleNamespace(
        cmd="build",
        project="project.yaml",
        profile="nightly",
        force=True,
        visuals=False,
        heartbeat_interval_seconds=None,
    )
    result = execute_command(
        args=args,
        plugin_root=None,
        workspace_context=None,
        cli_level_arg="DEBUG",
        cli_log_outputs=[],
    )

    assert result is None
    assert captured["profile_name"] == "nightly"
    assert captured["force"] is True
    assert captured["command_observability"] == CommandObservability(
        visuals=False,
        log_level="DEBUG",
    )


def test_execute_inspect_passes_command_and_profile(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def _capture_request(**kwargs):
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(
        "jerrythomas.cli.commands.profile_runner.build_runtime_run_request",
        _capture_request,
    )
    monkeypatch.setattr(
        "jerrythomas.cli.commands.profile_runner.execute_profile_request",
        lambda request: None,
    )

    args = _inspect_args()
    args.profile = "report"
    result = execute_command(
        args=args,
        plugin_root=None,
        workspace_context=None,
        cli_level_arg="INFO",
        cli_log_outputs=[],
    )

    assert result is None
    assert captured["command"] == "inspect"
    assert captured["profile_name"] == "report"
    assert captured["command_observability"] == CommandObservability(
        visuals=True,
        log_level="INFO",
    )


def test_execute_inspect_skips_when_no_enabled_profiles(monkeypatch, caplog) -> None:
    monkeypatch.setattr(
        "jerrythomas.cli.commands.profile_runner.build_runtime_run_request",
        lambda **kwargs: None,
    )

    with caplog.at_level(
        logging.INFO, logger="jerrythomas.cli.commands.profile_runner"
    ):
        result = execute_command(
            args=_inspect_args(),
            plugin_root=None,
            workspace_context=None,
            cli_level_arg=None,
            cli_log_outputs=[],
        )

    assert result is None
    assert "No enabled inspect profiles; skipping inspect." in caplog.text


def test_profile_request_logs_expected_error_before_error_summary(monkeypatch) -> None:
    error = ProfileCommandError("invalid profile")
    error.add_note("failed to finalize run")
    order: list[tuple[str, object]] = []

    def fail(_request) -> None:
        raise error

    monkeypatch.setattr(
        "jerrythomas.cli.commands.profile_runner.run_profiles",
        fail,
    )
    monkeypatch.setattr(
        "jerrythomas.cli.commands.profile_runner.logger.error",
        lambda message, *args: order.append(("log", message % args)),
    )
    monkeypatch.setattr(
        "jerrythomas.cli.commands.profile_runner.route_execution_event",
        lambda event, _logger: order.append(("event", event)),
    )
    monkeypatch.setattr(
        "jerrythomas.cli.commands.profile_runner.visual_summary",
        _noop_visual_summary,
    )
    times = iter((10.0, 11.5))
    monkeypatch.setattr(
        "jerrythomas.cli.commands.profile_runner.time.perf_counter",
        lambda: next(times),
    )

    with pytest.raises(SystemExit) as raised:
        execute_profile_request(_runtime_request())

    assert raised.value.code == 2
    assert raised.value.__cause__ is error
    assert order == [
        ("log", "invalid profile"),
        ("log", "failed to finalize run"),
        ("event", CommandFinished("serve", "error", 1.5)),
    ]


def test_profile_request_emits_one_success_summary(monkeypatch) -> None:
    events: list[CommandFinished] = []
    monkeypatch.setattr(
        "jerrythomas.cli.commands.profile_runner.run_profiles",
        lambda _request: None,
    )
    monkeypatch.setattr(
        "jerrythomas.cli.commands.profile_runner.route_execution_event",
        lambda event, _logger: events.append(event),
    )
    monkeypatch.setattr(
        "jerrythomas.cli.commands.profile_runner.visual_summary",
        _noop_visual_summary,
    )
    times = iter((3.0, 5.0))
    monkeypatch.setattr(
        "jerrythomas.cli.commands.profile_runner.time.perf_counter",
        lambda: next(times),
    )

    execute_profile_request(_runtime_request())

    assert events == [CommandFinished("serve", "success", 2.0)]


def test_profile_request_preserves_unexpected_failure(monkeypatch) -> None:
    error = RuntimeError("broken execution")
    events: list[CommandFinished] = []

    def fail(_request) -> None:
        raise error

    monkeypatch.setattr(
        "jerrythomas.cli.commands.profile_runner.run_profiles",
        fail,
    )
    monkeypatch.setattr(
        "jerrythomas.cli.commands.profile_runner.route_execution_event",
        lambda event, _logger: events.append(event),
    )
    monkeypatch.setattr(
        "jerrythomas.cli.commands.profile_runner.visual_summary",
        _noop_visual_summary,
    )
    times = iter((6.0, 7.0))
    monkeypatch.setattr(
        "jerrythomas.cli.commands.profile_runner.time.perf_counter",
        lambda: next(times),
    )

    with pytest.raises(RuntimeError) as raised:
        execute_profile_request(_runtime_request())

    assert raised.value is error
    assert events == [CommandFinished("serve", "error", 1.0)]


def test_profile_request_summary_failure_does_not_replace_command_failure(
    monkeypatch,
) -> None:
    command_error = RuntimeError("broken execution")

    def fail_command(_request) -> None:
        raise command_error

    def fail_summary(_event, _logger) -> None:
        raise OSError("broken reporter")

    monkeypatch.setattr(
        "jerrythomas.cli.commands.profile_runner.run_profiles",
        fail_command,
    )
    monkeypatch.setattr(
        "jerrythomas.cli.commands.profile_runner.route_execution_event",
        fail_summary,
    )
    monkeypatch.setattr(
        "jerrythomas.cli.commands.profile_runner.visual_summary",
        _noop_visual_summary,
    )

    with pytest.raises(RuntimeError) as raised:
        execute_profile_request(_runtime_request())

    assert raised.value is command_error
    assert raised.value.__notes__ == [
        "Reporting command completion also failed: broken reporter"
    ]


@pytest.mark.parametrize("error", [KeyboardInterrupt(), SystemExit(7)])
def test_profile_request_preserves_process_control_exceptions(
    monkeypatch,
    error,
) -> None:
    events: list[CommandFinished] = []
    messages: list[str] = []
    error.add_note("failed to finalize run")

    def fail(_request) -> None:
        raise error

    monkeypatch.setattr(
        "jerrythomas.cli.commands.profile_runner.run_profiles",
        fail,
    )
    monkeypatch.setattr(
        "jerrythomas.cli.commands.profile_runner.route_execution_event",
        lambda event, _logger: events.append(event),
    )
    monkeypatch.setattr(
        "jerrythomas.cli.commands.profile_runner.logger.error",
        lambda message, *args: messages.append(message % args),
    )
    monkeypatch.setattr(
        "jerrythomas.cli.commands.profile_runner.visual_summary",
        _noop_visual_summary,
    )
    times = iter((1.0, 1.25))
    monkeypatch.setattr(
        "jerrythomas.cli.commands.profile_runner.time.perf_counter",
        lambda: next(times),
    )

    with pytest.raises(type(error)) as raised:
        execute_profile_request(_runtime_request())

    assert raised.value is error
    assert messages == ["failed to finalize run"]
    assert events == [CommandFinished("serve", "error", 0.25)]


@pytest.mark.parametrize(
    "profile_request",
    (_runtime_request(visuals=True), _build_run_request(visuals=True)),
    ids=("runtime", "build"),
)
def test_profile_request_routes_summary_inside_enabled_visuals(
    monkeypatch,
    profile_request,
) -> None:
    order: list[object] = []

    @contextmanager
    def visual_summary(_level, enabled):
        order.append(("enter", enabled))
        try:
            yield
        finally:
            order.append(("exit", enabled))

    monkeypatch.setattr(
        "jerrythomas.cli.commands.profile_runner.run_profiles",
        lambda _request: None,
    )
    monkeypatch.setattr(
        "jerrythomas.cli.commands.profile_runner.route_execution_event",
        lambda event, _logger: order.append(event),
    )
    monkeypatch.setattr(
        "jerrythomas.cli.commands.profile_runner.visual_summary",
        visual_summary,
    )
    times = iter((2.0, 3.0))
    monkeypatch.setattr(
        "jerrythomas.cli.commands.profile_runner.time.perf_counter",
        lambda: next(times),
    )

    execute_profile_request(profile_request)

    assert order == [
        ("enter", True),
        CommandFinished(profile_request.command, "success", 1.0),
        ("exit", True),
    ]


def test_serve_result_json_serializes_completed_runs(monkeypatch, tmp_path, capsys):
    import json

    from jerrythomas.io.runs import (
        get_run_paths,
        start_run,
        finish_run_success,
        load_run,
    )

    request = _runtime_request()
    request.jobs = []
    request.artifact_settings.observability.log_output = SimpleNamespace(outputs=())
    paths = get_run_paths(tmp_path / "served", "run-1")
    other = get_run_paths(tmp_path / "other", "run-2")
    start_run(paths)
    finish_run_success(paths)
    start_run(other, preview="samples")
    finish_run_success(other)
    results = (load_run(paths.run_root), load_run(other.run_root))
    monkeypatch.setattr(
        "jerrythomas.cli.commands.profile_runner.build_runtime_run_request",
        lambda **_kwargs: request,
    )
    monkeypatch.setattr(
        "jerrythomas.cli.commands.profile_runner.execute_profile_request",
        lambda received: (
            results if received is request else pytest.fail("wrong request")
        ),
    )
    args = _serve_args()
    args.result_json = True
    execute_command(args, None, None, None, [])
    payload = json.loads(capsys.readouterr().out)
    assert payload["schema_version"] == 2
    assert [run["run_id"] for run in payload["runs"]] == ["run-1", "run-2"]
    assert [run["preview"] for run in payload["runs"]] == [None, "samples"]
    for run, result in zip(payload["runs"], results):
        assert run.pop("receipt") == str(result.metadata_path)
        assert run == result.metadata.model_dump(mode="json")


@pytest.mark.parametrize("stdout_use", ["data", "logs"])
def test_serve_result_json_rejects_stdout_conflicts_before_execution(
    monkeypatch, stdout_use
):
    request = _runtime_request()
    request.jobs = [
        SimpleNamespace(
            output=SimpleNamespace(
                transport="stdout" if stdout_use == "data" else "fs"
            ),
            observability=SimpleNamespace(log_output=SimpleNamespace(outputs=())),
        )
    ]
    request.artifact_settings.observability.log_output = SimpleNamespace(
        outputs=(
            SimpleNamespace(transport="stdout" if stdout_use == "logs" else "stderr"),
        )
    )
    monkeypatch.setattr(
        "jerrythomas.cli.commands.profile_runner.build_runtime_run_request",
        lambda **_kwargs: request,
    )
    monkeypatch.setattr(
        "jerrythomas.cli.commands.profile_runner.execute_profile_request",
        lambda _request: pytest.fail("must fail before execution"),
    )
    args = _serve_args()
    args.result_json = True
    with pytest.raises(ProfileCommandError, match="--result-json"):
        execute_command(args, None, None, None, [])


def test_serve_result_json_returns_empty_list_when_no_profiles_enabled(
    monkeypatch, capsys
):
    import json

    monkeypatch.setattr(
        "jerrythomas.cli.commands.profile_runner.build_runtime_run_request",
        lambda **_kwargs: None,
    )
    args = _serve_args()
    args.result_json = True
    execute_command(args, None, None, None, [])
    assert json.loads(capsys.readouterr().out) == {"schema_version": 2, "runs": []}


def test_serve_result_json_is_not_emitted_when_execution_fails(monkeypatch, capsys):
    request = _runtime_request()
    request.jobs = []
    request.artifact_settings.observability.log_output = SimpleNamespace(outputs=())
    monkeypatch.setattr(
        "jerrythomas.cli.commands.profile_runner.build_runtime_run_request",
        lambda **_kwargs: request,
    )

    def fail(_request):
        raise OSError("publication failed")

    monkeypatch.setattr(
        "jerrythomas.cli.commands.profile_runner.execute_profile_request", fail
    )
    args = _serve_args()
    args.result_json = True
    with pytest.raises(OSError, match="publication failed"):
        execute_command(args, None, None, None, [])
    assert capsys.readouterr().out == ""
