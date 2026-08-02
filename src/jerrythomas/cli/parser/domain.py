import argparse


def add_domain_command(sub, common: argparse.ArgumentParser) -> None:
    parser = sub.add_parser(
        "domain",
        help="create domains",
        parents=[common],
    )
    domain_sub = parser.add_subparsers(required=True)
    create = domain_sub.add_parser(
        "create",
        help="create a domain",
        description="Create a time-aware domain package rooted in TemporalRecord.",
    )
    create.add_argument("domain_name", help="domain name")
