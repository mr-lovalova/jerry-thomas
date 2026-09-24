import argparse

from jerrythomas.config.options import (
    OUTPUT_INSPECT_FORMATS,
    OUTPUT_TRANSPORTS,
    OUTPUT_VIEWS,
)
from jerrythomas.io.compression import COMPRESSION_CHOICES

from .common import (
    add_artifact_mode_flag,
    add_execution_observability_flags,
    add_project_flag,
    positive_integer,
)


def add_inspect_command(sub, common: argparse.ArgumentParser) -> None:
    parser = sub.add_parser(
        "inspect",
        help="run inspect operations through inspect profiles",
        parents=[common],
    )
    add_project_flag(parser)
    parser.add_argument(
        "--profile",
        help="select an inspect profile by name; explicitly selected disabled profiles still run",
    )
    parser.add_argument(
        "--limit",
        "-n",
        type=positive_integer,
        default=None,
        help="sample cap for matrix and custom runtime operations; coverage does not support it",
    )
    parser.add_argument(
        "--output-transport",
        choices=OUTPUT_TRANSPORTS,
        help="optional output transport override (stdout or fs) for inspect profiles",
    )
    parser.add_argument(
        "--output-format",
        choices=OUTPUT_INSPECT_FORMATS,
        help="optional output format override (jsonl/csv/pickle/txt/html) for inspect profiles",
    )
    parser.add_argument(
        "--output-directory",
        help="destination directory when using fs transport",
    )
    parser.add_argument(
        "--output-encoding",
        help="text encoding for fs jsonl/csv/txt outputs (default: utf-8)",
    )
    parser.add_argument(
        "--output-compression",
        choices=COMPRESSION_CHOICES,
        help="compress fs jsonl/csv output with gzip",
    )
    parser.add_argument(
        "--output-view",
        choices=OUTPUT_VIEWS,
        help="output representation view (jsonl: raw|flat, csv: flat, pickle: raw)",
    )
    add_artifact_mode_flag(parser)
    add_execution_observability_flags(parser)
