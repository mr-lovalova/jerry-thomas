import argparse

from jerrythomas.cli.version import short_version
from jerrythomas.cli.parser.build import add_build_command
from jerrythomas.cli.parser.clean import add_clean_command
from jerrythomas.cli.parser.common import build_logging_parent
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


def build_parser() -> argparse.ArgumentParser:
    root_common = build_logging_parent()
    command_common = build_logging_parent(suppress_defaults=True)
    parser = argparse.ArgumentParser(
        prog="jerry",
        description="Mixology-themed CLI for building and serving data pipelines.",
        parents=[root_common],
    )
    parser.add_argument(
        "--version",
        action="version",
        version=short_version(),
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("version", help="show installed Jerry version")
    sub.add_parser("env", help="show installed Jerry environment details")
    add_serve_command(sub, common=command_common)
    add_inspect_command(sub, common=command_common)
    add_build_command(sub, common=command_common)
    add_materialize_command(sub, common=command_common)
    add_clean_command(sub, common=command_common)
    add_demo_command(sub, common=command_common)
    add_list_command(sub, common=command_common)
    add_source_command(sub, common=command_common)
    add_domain_command(sub, common=command_common)
    add_simple_scaffold_command(
        sub,
        common=command_common,
        cmd="dto",
        help_text="create DTOs",
        arg_help="DTO class name",
    )
    add_simple_scaffold_command(
        sub,
        common=command_common,
        cmd="parser",
        help_text="create parsers",
        arg_help="Parser class name",
    )
    add_simple_scaffold_command(
        sub,
        common=command_common,
        cmd="mapper",
        help_text="create mappers",
        arg_help="Mapper function name",
    )
    add_simple_scaffold_command(
        sub,
        common=command_common,
        cmd="loader",
        help_text="create loaders",
        arg_help="Loader name",
    )
    add_inflow_command(sub, common=command_common)
    add_stream_command(sub, common=command_common)
    add_plugin_command(sub, common=command_common)
    return parser
