from .common import add_logging_flags


def add_domain_command(sub) -> None:
    parser = sub.add_parser(
        "domain",
        help="create domains",
    )
    domain_sub = parser.add_subparsers(required=True)
    create = domain_sub.add_parser(
        "create",
        help="create a domain",
        description="Create a time-aware domain package rooted in TemporalRecord.",
    )
    create.add_argument("domain_name", help="domain name")
    add_logging_flags(create)
    add_logging_flags(parser)
