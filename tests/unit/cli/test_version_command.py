from importlib.metadata import EntryPoint
from pathlib import Path

import pytest

from jerrythomas.cli.command_router import execute_command
from jerrythomas.cli.parser_builder import build_parser
from jerrythomas.cli.version import short_version, version_report
from jerrythomas.plugins import PARSERS_EP, PluginDistribution


def test_root_version_flag_prints_short_version(capsys) -> None:
    parser = build_parser()

    with pytest.raises(SystemExit) as exc:
        parser.parse_args(["--version"])

    assert exc.value.code == 0
    assert capsys.readouterr().out.strip() == short_version()


def test_version_command_prints_short_version(capsys) -> None:
    args = build_parser().parse_args(["version"])

    result = execute_command(
        args,
        plugin_root=None,
        workspace_context=None,
        cli_level_arg=None,
        cli_log_outputs=[],
    )

    assert result is None
    assert capsys.readouterr().out.strip() == short_version()


def test_env_command_prints_diagnostic_report(capsys) -> None:
    args = build_parser().parse_args(["env"])

    result = execute_command(
        args,
        plugin_root=None,
        workspace_context=None,
        cli_level_arg=None,
        cli_log_outputs=[],
    )

    assert result is None
    assert capsys.readouterr().out.strip() == version_report()


def test_env_reports_plugin_provider_version_target_and_editable_path(monkeypatch):
    monkeypatch.setattr(
        "jerrythomas.cli.version.plugin_distributions",
        lambda: (
            PluginDistribution(
                name="research-plugins",
                version="2.3.0",
                editable_path=Path("/workspace/plugins"),
                entrypoints=(
                    EntryPoint(
                        name="custom_csv", group=PARSERS_EP, value="research.csv:parse"
                    ),
                ),
            ),
        ),
    )
    report = version_report()
    assert "installed plugin providers:" in report
    assert "research-plugins 2.3.0" in report
    assert "editable: /workspace/plugins" in report
    assert "jerrythomas.parsers/custom_csv -> research.csv:parse" in report
