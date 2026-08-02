import argparse


def add_stream_command(sub, common: argparse.ArgumentParser) -> None:
    parser = sub.add_parser(
        "stream",
        help="manage source-backed, aligned, and broadcast streams",
        parents=[common],
    )
    stream_sub = parser.add_subparsers(dest="stream_cmd", required=True)
    create = stream_sub.add_parser("create", help="create a stream")
    create.add_argument(
        "--identity",
        action="store_true",
        help="use the built-in identity mapper without prompting",
    )
