import argparse

from .common import (
    add_dataset_flag,
    add_execution_observability_flags,
    add_project_flag,
)


def add_build_command(sub, common: argparse.ArgumentParser) -> None:
    parser = sub.add_parser(
        "build",
        help="materialize project artifacts (metadata, statistics, etc.)",
        parents=[common],
    )
    add_dataset_flag(parser)
    add_project_flag(parser)
    parser.add_argument(
        "--force",
        action="store_true",
        help="rebuild even when the artifact hash matches the last run",
    )
    parser.add_argument(
        "--profile",
        help="select a build profile by name; explicitly selected disabled profiles still run",
    )
    add_execution_observability_flags(parser)
