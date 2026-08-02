import argparse

from jerrythomas.cli.source_options import SOURCE_FORMATS, SOURCE_TRANSPORTS


def add_source_command(sub, common: argparse.ArgumentParser) -> None:
    parser = sub.add_parser(
        "source",
        help="create raw sources",
        parents=[common],
    )
    source_sub = parser.add_subparsers(required=True)
    create = source_sub.add_parser(
        "create",
        help="create a provider+dataset source (yaml only)",
        description=(
            "Create a source YAML using transport + format or a loader entrypoint.\n\n"
            "Usage:\n"
            "  jerry source create <provider>.<dataset> -t fs -f csv\n"
            "  jerry source create <provider>.<dataset> -t http -f json\n"
            "  jerry source create <provider>.<dataset> -t synthetic\n\n"
            "  jerry source create <provider>.<dataset> --loader mypkg.weather\n"
            "  jerry source create <provider>.<dataset> --parser myparser\n\n"
            "Examples:\n"
            "  fs CSV:        -t fs  -f csv\n"
            "  fs NDJSON:     -t fs  -f jsonl\n"
            "  fs Parquet:    -t fs  -f parquet\n"
            "  HTTP JSON:     -t http -f json\n"
            "  Synthetic:     -t synthetic"
        ),
    )
    create.add_argument(
        "source_id",
        nargs="?",
        help="source id in provider.dataset form",
    )
    create.add_argument(
        "--transport",
        "-t",
        choices=SOURCE_TRANSPORTS,
        required=False,
        help="how data is accessed: fs/http/synthetic",
    )
    create.add_argument(
        "--format",
        "-f",
        choices=SOURCE_FORMATS,
        help="data format for fs/http transports",
    )
    create.add_argument(
        "--loader",
        help="loader entrypoint (overrides --transport/--format)",
    )
    create.add_argument(
        "--parser",
        help="parser entrypoint (defaults to identity)",
    )
