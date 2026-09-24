from .common import add_project_flag, add_logging_flags


def add_stream_command(sub) -> None:
    parser = sub.add_parser(
        "stream",
        help="manage source-backed, aligned, and broadcast streams",
    )
    stream_sub = parser.add_subparsers(dest="stream_cmd", required=True)
    create = stream_sub.add_parser("create", help="create a stream")
    add_project_flag(create)
    create.add_argument(
        "--identity",
        action="store_true",
        help="use the built-in identity mapper without prompting",
    )
    add_logging_flags(create)
    add_logging_flags(parser)
