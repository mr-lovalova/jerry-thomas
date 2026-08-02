import pytest

from jerrythomas.cli.command_router import execute_command
from jerrythomas.cli.parser_builder import build_parser


def test_clean_parser_defaults_to_dry_run() -> None:
    args = build_parser().parse_args(["clean"])

    assert args.cmd == "clean"
    assert args.yes is False
    assert args.older_than == "0h"


def test_clean_command_dispatches(monkeypatch) -> None:
    calls = {}

    def handle_clean(*, yes, older_than):
        calls["yes"] = yes
        calls["older_than"] = older_than

    monkeypatch.setattr(
        "jerrythomas.cli.command_router.handle_clean",
        handle_clean,
    )
    args = build_parser().parse_args(["clean", "--yes", "--older-than", "24h"])

    result = execute_command(
        args=args,
        plugin_root=None,
        workspace_context=None,
        cli_level_arg=None,
        cli_log_outputs=[],
    )

    assert result is None
    assert calls == {"yes": True, "older_than": "24h"}


@pytest.mark.parametrize("age", ["soon", "inf"])
def test_clean_command_rejects_invalid_age(age: str) -> None:
    args = build_parser().parse_args(["clean", "--older-than", age])

    with pytest.raises(SystemExit, match="age must"):
        execute_command(
            args=args,
            plugin_root=None,
            workspace_context=None,
            cli_level_arg=None,
            cli_log_outputs=[],
        )
