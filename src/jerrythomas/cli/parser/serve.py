import argparse

from jerrythomas.config.options import OUTPUT_FORMATS, OUTPUT_TRANSPORTS, OUTPUT_VIEWS
from jerrythomas.config.preview import PREVIEW_STAGES
from jerrythomas.io.compression import COMPRESSION_CHOICES

from .common import (
    add_artifact_mode_flag,
    add_dataset_flag,
    add_execution_observability_flags,
    add_project_flag,
    positive_integer,
)


def add_serve_command(sub, common: argparse.ArgumentParser) -> None:
    parser = sub.add_parser(
        "serve",
        help="produce dataset samples with configurable logging",
        parents=[common],
    )
    add_dataset_flag(parser)
    add_project_flag(parser)
    parser.add_argument(
        "--result-json",
        action="store_true",
        help="write completed run results as JSON to stdout; requires filesystem data outputs",
    )
    parser.add_argument(
        "--limit",
        "-n",
        type=positive_integer,
        default=None,
        help="optional cap on the number of samples to emit",
    )
    parser.add_argument(
        "--output-transport",
        choices=OUTPUT_TRANSPORTS,
        help="output transport (stdout or fs) for serve runs",
    )
    parser.add_argument(
        "--output-format",
        choices=OUTPUT_FORMATS,
        help="output format (jsonl/csv/parquet/pickle) for serve runs",
    )
    parser.add_argument(
        "--output-directory",
        help="destination directory when using fs transport",
    )
    parser.add_argument(
        "--output-encoding",
        help="text encoding for fs jsonl/csv outputs (default: utf-8)",
    )
    parser.add_argument(
        "--output-compression",
        choices=COMPRESSION_CHOICES,
        help="compress fs jsonl/csv output with gzip",
    )
    parser.add_argument(
        "--output-view",
        choices=OUTPUT_VIEWS,
        help=(
            "output representation view "
            "(jsonl: raw|flat, csv/parquet: flat, pickle: raw)"
        ),
    )
    parser.add_argument(
        "--profile",
        help="select a serve profile by name; explicitly selected disabled profiles still run",
    )
    parser.add_argument(
        "--preview",
        choices=PREVIEW_STAGES,
        default=None,
        help="stop serve after a semantic pipeline stage",
    )
    add_execution_observability_flags(parser)
    add_artifact_mode_flag(parser)
