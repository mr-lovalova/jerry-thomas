from .common import add_project_flag, add_logging_flags


def add_inflow_command(sub) -> None:
    parser = sub.add_parser(
        "inflow",
        help="create end-to-end inflow scaffolds",
    )
    inflow_sub = parser.add_subparsers(dest="inflow_cmd", required=True)
    create = inflow_sub.add_parser("create", help="create an inflow")
    add_project_flag(create)
    add_logging_flags(create)
    add_logging_flags(parser)
