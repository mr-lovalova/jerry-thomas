import argparse
import math
from typing import Sequence

from jerrythomas.config.options import OUTPUT_TRANSPORTS, OUTPUT_VIEWS
from jerrythomas.io.compression import COMPRESSION_CHOICES

from jerrythomas.config.profiles.build import ARTIFACT_MODES


def _heartbeat_interval_seconds(value: str) -> float:
    try:
        interval = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "--heartbeat-interval must be a non-negative number of seconds"
        ) from exc
    if not math.isfinite(interval) or interval < 0:
        raise argparse.ArgumentTypeError(
            "--heartbeat-interval must be a finite, non-negative number of seconds"
        )
    return interval


def positive_integer(value: str) -> int:
    try:
        number = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("value must be a positive integer") from exc
    if number < 1:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return number


def add_project_flag(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--project",
        "-p",
        default=None,
        help="project alias, folder, or project.yaml path",
    )


def add_artifact_mode_flag(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--artifact-mode",
        choices=ARTIFACT_MODES,
        default=None,
        help="artifact policy: auto | rebuild | require_current",
    )


def add_execution_observability_flags(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--visuals",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="enable terminal visuals (default: enabled)",
    )
    parser.add_argument(
        "--heartbeat-interval",
        dest="heartbeat_interval_seconds",
        type=_heartbeat_interval_seconds,
        default=None,
        metavar="SECONDS",
        help="pipeline heartbeat interval in seconds; set to 0 to disable",
    )


def add_logging_flags(
    parser: argparse.ArgumentParser, *, suppress_defaults: bool = True
) -> None:
    default = argparse.SUPPRESS if suppress_defaults else None
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        type=str.upper,
        default=default,
        help="set logging level (default: INFO)",
    )
    parser.add_argument(
        "--log-output",
        action="append",
        metavar="TARGET",
        default=default,
        help="repeatable log output target: stderr | stdout | fs:<path> | execution[:<relative-path>]",
    )


def add_profile_flag(parser: argparse.ArgumentParser, command: str) -> None:
    parser.add_argument(
        "--profile",
        help=f"select a {command} profile by name; explicitly selected disabled profiles still run",
    )


def add_runtime_output_flags(
    parser: argparse.ArgumentParser, *, formats: Sequence[str]
) -> None:
    parser.add_argument(
        "--output-transport",
        choices=OUTPUT_TRANSPORTS,
        help="override the profile output transport (stdout or fs)",
    )
    parser.add_argument(
        "--output-format",
        choices=formats,
        help="override the profile output format",
    )
    parser.add_argument(
        "--output-directory",
        help="destination directory when using fs transport",
    )
    parser.add_argument(
        "--output-encoding",
        help="text encoding for filesystem text outputs (default: utf-8)",
    )
    parser.add_argument(
        "--output-compression",
        choices=COMPRESSION_CHOICES,
        help="compress fs jsonl/csv output with gzip",
    )
    parser.add_argument(
        "--output-view",
        choices=OUTPUT_VIEWS,
        help="record representation; supported views depend on the output format",
    )
