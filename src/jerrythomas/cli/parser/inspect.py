from jerrythomas.config.options import OUTPUT_INSPECT_FORMATS

from .common import (
    add_artifact_mode_flag,
    add_execution_observability_flags,
    add_logging_flags,
    add_profile_flag,
    add_project_flag,
    add_runtime_output_flags,
    positive_integer,
)


def add_inspect_command(sub) -> None:
    parser = sub.add_parser(
        "inspect",
        help="run inspect operations through inspect profiles",
    )
    add_project_flag(parser)
    add_profile_flag(parser, "inspect")
    parser.add_argument(
        "--limit",
        "-n",
        type=positive_integer,
        default=None,
        help="sample cap for matrix and custom output operations; coverage does not support it",
    )
    add_runtime_output_flags(parser, formats=OUTPUT_INSPECT_FORMATS)
    add_artifact_mode_flag(parser)
    add_logging_flags(parser)
    add_execution_observability_flags(parser)
