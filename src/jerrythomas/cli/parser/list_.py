import argparse


def add_list_command(sub, common: argparse.ArgumentParser) -> None:
    parser = sub.add_parser(
        "list",
        help="list known resources",
        parents=[common],
    )
    list_sub = parser.add_subparsers(dest="list_cmd", required=True)
    list_sub.add_parser("sources", help="list sources")
    list_sub.add_parser("domains", help="list domains")
    list_sub.add_parser("parsers", help="list parsers")
    list_sub.add_parser("dtos", help="list DTOs")
    list_sub.add_parser("mappers", help="list mappers")
    list_sub.add_parser("combiners", help="list combiners")
    list_sub.add_parser("loaders", help="list loaders")
