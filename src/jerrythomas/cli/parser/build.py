from .common import (
    add_artifact_mode_flag,
    add_execution_observability_flags,
    add_logging_flags,
    add_profile_flag,
    add_project_flag,
)


def add_build_command(sub) -> None:
    parser = sub.add_parser(
        "build",
        help="build configured artifact operations through build profiles",
    )
    add_project_flag(parser)
    add_profile_flag(parser, "build")
    add_artifact_mode_flag(parser)
    add_logging_flags(parser)
    add_execution_observability_flags(parser)
