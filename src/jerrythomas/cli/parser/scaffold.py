import argparse


def add_simple_scaffold_command(
    sub,
    common: argparse.ArgumentParser,
    cmd: str,
    help_text: str,
    arg_help: str,
    subcmd: str = "create",
) -> None:
    parser = sub.add_parser(cmd, help=help_text, parents=[common])
    cmd_sub = parser.add_subparsers(dest=f"{cmd}_cmd", required=True)
    create = cmd_sub.add_parser(subcmd, help=f"create a {cmd}")
    create.add_argument("name", nargs="?", help=arg_help)
