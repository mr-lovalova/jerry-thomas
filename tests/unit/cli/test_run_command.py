import logging
from pathlib import Path
from types import SimpleNamespace

import pytest

from datapipeline.cli.command_router import execute_command
from datapipeline.cli.output_options import build_cli_output_config
from datapipeline.cli.parser_builder import build_parser
from datapipeline.config.profiles.output import ServeOutputConfig


def _serve_args() -> SimpleNamespace:
    return SimpleNamespace(
        cmd="serve",
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
        visuals="on",
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
        visuals="on",
        heartbeat_interval_seconds=None,
    )


def test_build_cli_output_config_fs_requires_directory() -> None:
    with pytest.raises(SystemExit) as exc:
        build_cli_output_config(
            transport="fs",
            fmt="jsonl",
            directory=None,
        )
    assert exc.value.code == 2


def test_build_cli_output_config_rejects_unknown_transport() -> None:
    with pytest.raises(SystemExit) as exc:
        build_cli_output_config(
            transport="s3",
            fmt="jsonl",
            directory=None,
        )

    assert exc.value.code == 2


def test_build_cli_output_config_stdout_rejects_directory() -> None:
    with pytest.raises(SystemExit) as exc:
        build_cli_output_config(
            transport="stdout",
            fmt="jsonl",
            directory="out",
        )
    assert exc.value.code == 2


def test_build_cli_output_config_rejects_stdout_csv() -> None:
    with pytest.raises(SystemExit) as exc:
        build_cli_output_config(
            transport="stdout",
            fmt="csv",
            directory=None,
            view="flat",
        )
    assert exc.value.code == 2


def test_build_cli_output_config_fs_populates_output() -> None:
    config = build_cli_output_config(
        transport="fs",
        fmt="jsonl",
        directory="artifacts",
    )
    assert config is not None
    assert config.transport == "fs"
    assert config.format == "jsonl"
    assert config.view is None
    assert config.encoding == "utf-8"
    assert config.directory == Path("artifacts").resolve()


def test_build_cli_output_config_fs_populates_gzip_compression() -> None:
    config = build_cli_output_config(
        transport="fs",
        fmt="jsonl",
        directory="artifacts",
        output_compression="gzip",
    )

    assert config is not None
    assert config.compression == "gzip"


def test_output_compression_requires_complete_output_override() -> None:
    with pytest.raises(SystemExit) as exc:
        build_cli_output_config(
            transport=None,
            fmt=None,
            directory=None,
            output_compression="gzip",
        )

    assert exc.value.code == 2


def test_build_cli_output_config_rejects_stdout_compression() -> None:
    with pytest.raises(SystemExit) as exc:
        build_cli_output_config(
            transport="stdout",
            fmt="jsonl",
            directory=None,
            output_compression="gzip",
        )

    assert exc.value.code == 2


def test_build_cli_output_config_honors_view() -> None:
    config = build_cli_output_config(
        transport="stdout",
        fmt="jsonl",
        directory=None,
        view="flat",
    )
    assert config is not None
    assert config.view == "flat"


def test_build_cli_output_config_rejects_non_flat_csv_view() -> None:
    with pytest.raises(SystemExit) as exc:
        build_cli_output_config(
            transport="fs",
            fmt="csv",
            directory="out",
            view="raw",
        )
    assert exc.value.code == 2


def test_build_cli_output_config_rejects_non_raw_pickle_view() -> None:
    with pytest.raises(SystemExit) as exc:
        build_cli_output_config(
            transport="fs",
            fmt="pickle",
            directory="out",
            view="flat",
        )
    assert exc.value.code == 2


def test_build_cli_output_config_honors_encoding_for_fs() -> None:
    config = build_cli_output_config(
        transport="fs",
        fmt="csv",
        directory="out",
        output_encoding="utf-8-sig",
        view="flat",
    )
    assert config is not None
    assert config.encoding == "utf-8-sig"


def test_build_cli_output_config_rejects_encoding_for_stdout() -> None:
    with pytest.raises(SystemExit) as exc:
        build_cli_output_config(
            transport="stdout",
            fmt="jsonl",
            directory=None,
            output_encoding="utf-8",
        )
    assert exc.value.code == 2


def test_build_cli_output_config_rejects_unknown_encoding() -> None:
    with pytest.raises(SystemExit) as exc:
        build_cli_output_config(
            transport="fs",
            fmt="jsonl",
            directory="out",
            output_encoding="definitely-not-a-codec",
        )
    assert exc.value.code == 2


def test_execute_serve_propagates_keyboard_interrupt(monkeypatch) -> None:
    monkeypatch.setattr(
        "datapipeline.cli.commands.profile_runner.build_runtime_run_request",
        lambda **kwargs: object(),
    )

    calls = {"run_profiles": 0}

    def _interrupting_execute(request):
        calls["run_profiles"] += 1
        raise KeyboardInterrupt()

    monkeypatch.setattr(
        "datapipeline.cli.commands.profile_runner.run_profiles",
        _interrupting_execute,
    )

    with pytest.raises(KeyboardInterrupt):
        execute_command(
            args=_serve_args(),
            plugin_root=None,
            workspace_context=None,
            cli_level_arg=None,
            base_level_name="INFO",
            cli_log_outputs=[],
        )

    assert calls["run_profiles"] == 1


def test_execute_serve_runs_request_from_builder(monkeypatch) -> None:
    sentinel_request = object()
    captured: dict[str, object] = {}

    def _build_request(**kwargs):
        captured.update(kwargs)
        return sentinel_request

    monkeypatch.setattr(
        "datapipeline.cli.commands.profile_runner.build_runtime_run_request",
        _build_request,
    )

    seen = {"request": None}

    def _capture(request):
        seen["request"] = request

    monkeypatch.setattr(
        "datapipeline.cli.commands.profile_runner.run_profiles", _capture
    )

    args = _serve_args()
    args.artifact_mode = "FORCE"
    result = execute_command(
        args=args,
        plugin_root=None,
        workspace_context=None,
        cli_level_arg=None,
        base_level_name="INFO",
        cli_log_outputs=[],
    )

    assert result is None
    assert seen["request"] is sentinel_request
    assert captured["command"] == "serve"
    assert captured["artifact_mode"] == "FORCE"


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
        "datapipeline.cli.commands.profile_runner.build_runtime_run_request",
        _capture_request,
    )
    monkeypatch.setattr(
        "datapipeline.cli.commands.profile_runner.run_profiles",
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
        base_level_name="INFO",
        cli_log_outputs=[],
    )

    output = captured["cli_output"]
    assert isinstance(output, ServeOutputConfig)
    assert output.transport == "fs"
    assert output.format == "jsonl"
    assert output.directory == tmp_path
    assert output.compression == "gzip"


def test_execute_serve_skips_when_no_enabled_profiles(monkeypatch, caplog) -> None:
    monkeypatch.setattr(
        "datapipeline.cli.commands.profile_runner.build_runtime_run_request",
        lambda **kwargs: None,
    )

    with caplog.at_level(
        logging.INFO, logger="datapipeline.cli.commands.profile_runner"
    ):
        result = execute_command(
            args=_serve_args(),
            plugin_root=None,
            workspace_context=None,
            cli_level_arg=None,
            base_level_name="INFO",
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
        "datapipeline.cli.commands.profile_runner.build_build_run_request",
        _capture_request,
    )
    monkeypatch.setattr(
        "datapipeline.cli.commands.profile_runner.run_profiles",
        lambda request: None,
    )

    args = SimpleNamespace(
        cmd="build",
        project="project.yaml",
        profile="nightly",
        force=True,
        visuals="off",
        heartbeat_interval_seconds=None,
    )
    result = execute_command(
        args=args,
        plugin_root=None,
        workspace_context=None,
        cli_level_arg="DEBUG",
        base_level_name="DEBUG",
        cli_log_outputs=[],
    )

    assert result is None
    assert captured["profile_name"] == "nightly"
    assert captured["force"] is True


def test_execute_inspect_passes_command_and_profile(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def _capture_request(**kwargs):
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(
        "datapipeline.cli.commands.profile_runner.build_runtime_run_request",
        _capture_request,
    )
    monkeypatch.setattr(
        "datapipeline.cli.commands.profile_runner.run_profiles",
        lambda request: None,
    )

    args = _inspect_args()
    args.profile = "report"
    result = execute_command(
        args=args,
        plugin_root=None,
        workspace_context=None,
        cli_level_arg="INFO",
        base_level_name="INFO",
        cli_log_outputs=[],
    )

    assert result is None
    assert captured["command"] == "inspect"
    assert captured["profile_name"] == "report"


def test_execute_inspect_skips_when_no_enabled_profiles(monkeypatch, caplog) -> None:
    monkeypatch.setattr(
        "datapipeline.cli.commands.profile_runner.build_runtime_run_request",
        lambda **kwargs: None,
    )

    with caplog.at_level(
        logging.INFO, logger="datapipeline.cli.commands.profile_runner"
    ):
        result = execute_command(
            args=_inspect_args(),
            plugin_root=None,
            workspace_context=None,
            cli_level_arg=None,
            base_level_name="INFO",
            cli_log_outputs=[],
        )

    assert result is None
    assert "No enabled inspect profiles; skipping inspect." in caplog.text
