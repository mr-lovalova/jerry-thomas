import argparse

from .common import (
    add_artifact_mode_flag,
    add_execution_observability_flags,
    add_logging_flags,
    add_profile_flag,
    add_project_flag,
)


def add_export_command(sub) -> None:
    parser = sub.add_parser(
        "export",
        help="export reusable stream outputs",
        description="Run configured export profiles.",
    )
    add_project_flag(parser)
    add_profile_flag(parser, "export")
    parser.add_argument(
        "--output-file",
        dest="output",
        help="override one profile's destination .jsonl or .jsonl.gz file (requires --profile)",
    )
    parser.add_argument(
        "--overwrite",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="overwrite existing exported outputs",
    )
    parser.add_argument(
        "--result-json",
        action="store_true",
        help="write completed export results as JSON to stdout",
    )
    add_artifact_mode_flag(parser)
    add_logging_flags(parser)
    add_execution_observability_flags(parser)
