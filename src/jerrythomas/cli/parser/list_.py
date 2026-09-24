import argparse

from .common import add_project_flag


def add_list_command(sub, common: argparse.ArgumentParser) -> None:
    parser = sub.add_parser(
        "list",
        help="list known resources",
        parents=[common],
    )
    list_sub = parser.add_subparsers(dest="list_cmd", required=True)
    for resource in ("datasets", "streams", "profiles", "sources"):
        resource_parser = list_sub.add_parser(resource, help=f"list project {resource}")
        add_project_flag(resource_parser)
    list_sub.add_parser("domains", help="list domains")
    list_sub.add_parser("parsers", help="list parsers")
    list_sub.add_parser("dtos", help="list DTOs")
    list_sub.add_parser("mappers", help="list mappers")
    list_sub.add_parser("combiners", help="list combiners")
    list_sub.add_parser("loaders", help="list loaders")
