from .common import add_logging_flags, add_project_flag


def add_list_command(sub) -> None:
    parser = sub.add_parser(
        "list",
        help="list known resources",
    )
    list_sub = parser.add_subparsers(dest="list_cmd", required=True)
    for resource in ("datasets", "streams", "profiles", "sources"):
        resource_parser = list_sub.add_parser(resource, help=f"list project {resource}")
        add_project_flag(resource_parser)
        add_logging_flags(resource_parser)
    for resource in ("domains", "parsers", "dtos", "mappers", "combiners", "loaders"):
        resource_parser = list_sub.add_parser(resource, help=f"list {resource}")
        add_logging_flags(resource_parser)
    add_logging_flags(parser)
