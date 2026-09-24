from .common import add_logging_flags


def add_demo_command(sub) -> None:
    parser = sub.add_parser(
        "demo",
        help="create a self-contained demo workspace",
    )
    demo_sub = parser.add_subparsers(dest="demo_cmd", required=True)
    demo_create = demo_sub.add_parser(
        "create",
        help="create a standalone demo plugin named demo",
    )
    demo_create.add_argument(
        "--out",
        "-o",
        help="override parent directory (demo will be created inside)",
    )
    add_logging_flags(demo_create)
    add_logging_flags(parser)
