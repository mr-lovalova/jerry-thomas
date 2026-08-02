import argparse


def add_clean_command(sub, common: argparse.ArgumentParser) -> None:
    parser = sub.add_parser(
        "clean",
        parents=[common],
        help="inspect or remove stale Jerry sort spill directories",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="delete matching sort spill directories; default is dry-run",
    )
    parser.add_argument(
        "--older-than",
        default="0h",
        metavar="AGE",
        help="only include dirs older than AGE, e.g. 30m, 24h, 7d",
    )
