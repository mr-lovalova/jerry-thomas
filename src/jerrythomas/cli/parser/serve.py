from jerrythomas.config.options import OUTPUT_FORMATS
from jerrythomas.config.preview import PREVIEW_STAGES

from .common import (
    add_artifact_mode_flag,
    add_execution_observability_flags,
    add_logging_flags,
    add_profile_flag,
    add_project_flag,
    add_runtime_output_flags,
    positive_integer,
)


def add_serve_command(sub) -> None:
    parser = sub.add_parser(
        "serve",
        help="run dataset or stream operations through serve profiles",
    )
    add_project_flag(parser)
    add_profile_flag(parser, "serve")
    parser.add_argument(
        "--limit",
        "-n",
        type=positive_integer,
        default=None,
        help="optional cap on dataset samples or stream records per output",
    )
    parser.add_argument(
        "--preview",
        choices=PREVIEW_STAGES,
        default=None,
        help="stop serve after a semantic pipeline stage",
    )
    add_runtime_output_flags(parser, formats=OUTPUT_FORMATS)
    parser.add_argument(
        "--result-json",
        action="store_true",
        help="write completed run results as JSON to stdout; requires filesystem data outputs",
    )
    add_artifact_mode_flag(parser)
    add_logging_flags(parser)
    add_execution_observability_flags(parser)
