import argparse

from jerrythomas.cli.version import short_version
from jerrythomas.cli.parser.build import add_build_command
from jerrythomas.cli.parser.clean import add_clean_command
from jerrythomas.cli.parser.common import add_logging_flags
from jerrythomas.cli.parser.demo import add_demo_command
from jerrythomas.cli.parser.domain import add_domain_command
from jerrythomas.cli.parser.inflow import add_inflow_command
from jerrythomas.cli.parser.inspect import add_inspect_command
from jerrythomas.cli.parser.list_ import add_list_command
from jerrythomas.cli.parser.materialize import add_materialize_command
from jerrythomas.cli.parser.plugin import add_plugin_command
from jerrythomas.cli.parser.scaffold import add_simple_scaffold_command
from jerrythomas.cli.parser.serve import add_serve_command
from jerrythomas.cli.parser.source import add_source_command
from jerrythomas.cli.parser.stream import add_stream_command


class _ArgumentParser(argparse.ArgumentParser):
    def __init__(self, *args, **kwargs):
        kwargs["allow_abbrev"] = False
        super().__init__(*args, **kwargs)


def build_parser() -> argparse.ArgumentParser:
    parser = _ArgumentParser(
        prog="jerry",
        description="Mixology-themed CLI for building and serving data pipelines.",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=short_version(),
    )
    add_logging_flags(parser, suppress_defaults=False)
    sub = parser.add_subparsers(dest="cmd", required=True)

    for command, help_text in (
        ("version", "show installed Jerry version"),
        ("env", "show installed Jerry environment details"),
    ):
        add_logging_flags(sub.add_parser(command, help=help_text))
    add_serve_command(sub)
    add_inspect_command(sub)
    add_build_command(sub)
    add_materialize_command(sub)
    add_clean_command(sub)
    add_demo_command(sub)
    add_list_command(sub)
    add_source_command(sub)
    add_domain_command(sub)
    add_simple_scaffold_command(
        sub,
        cmd="dto",
        help_text="create DTOs",
        arg_help="DTO class name",
    )
    add_simple_scaffold_command(
        sub,
        cmd="parser",
        help_text="create parsers",
        arg_help="Parser class name",
    )
    add_simple_scaffold_command(
        sub,
        cmd="mapper",
        help_text="create mappers",
        arg_help="Mapper function name",
    )
    add_simple_scaffold_command(
        sub,
        cmd="loader",
        help_text="create loaders",
        arg_help="Loader name",
    )
    add_inflow_command(sub)
    add_stream_command(sub)
    add_plugin_command(sub)
    return parser
