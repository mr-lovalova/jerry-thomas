from .common import add_logging_flags


def add_plugin_command(sub) -> None:
    parser = sub.add_parser(
        "plugin",
        help="scaffold plugin workspaces",
    )
    plugin_sub = parser.add_subparsers(required=True)
    create = plugin_sub.add_parser("create", help="create a plugin skeleton")
    create.add_argument(
        "plugin_name",
        help="plugin distribution name",
    )
    create.add_argument("--out", "-o", default=".")
    add_logging_flags(create)
    add_logging_flags(parser)
