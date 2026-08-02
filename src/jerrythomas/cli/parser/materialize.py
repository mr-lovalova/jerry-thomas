import argparse

from .common import (
    add_artifact_mode_flag,
    add_dataset_flag,
    add_execution_observability_flags,
    add_project_flag,
)


def add_materialize_command(sub, common: argparse.ArgumentParser) -> None:
    parser = sub.add_parser(
        "materialize",
        help="materialize reusable stream outputs",
        description="Run configured materialize profiles.",
        parents=[common],
    )
    add_dataset_flag(parser)
    add_project_flag(parser)
    parser.add_argument(
        "--profile",
        help="select one materialize profile by name; explicitly selected disabled profiles still run",
    )
    parser.add_argument(
        "--output",
        help="override one profile's destination .jsonl or .jsonl.gz file (requires --profile)",
    )
    parser.add_argument(
        "--overwrite",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="overwrite existing materialized outputs",
    )
    add_artifact_mode_flag(parser)
    add_execution_observability_flags(parser)
