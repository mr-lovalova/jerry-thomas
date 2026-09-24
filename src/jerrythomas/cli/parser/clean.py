from .common import add_logging_flags


def add_clean_command(sub) -> None:
    parser = sub.add_parser(
        "clean",
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
    add_logging_flags(parser)
