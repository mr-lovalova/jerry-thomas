import logging

from jerrythomas.cli.commands.profile_runner import execute_profile_request
from jerrythomas.cli.logging_setup import configure_profile_logging
from jerrythomas.cli.run_results import validate_result_json_outputs, write_run_results
from jerrythomas.cli.workspace import WorkspaceContext
from jerrythomas.execution.settings import CommandObservability, LogOutputTarget
from jerrythomas.profiles.request_builder import build_materialize_run_request
from jerrythomas.services.path_policy import resolve_workspace_path

logger = logging.getLogger(__name__)


def handle(
    project: str,
    profile_name: str | None,
    output: str | None,
    overwrite: bool | None,
    artifact_mode: str | None,
    visuals: bool | None,
    heartbeat_interval_seconds: float | None,
    cli_log_level: str | None,
    cli_log_outputs: list[LogOutputTarget],
    workspace: WorkspaceContext | None,
    result_json: bool = False,
) -> None:
    if profile_name is None and output is not None:
        logger.error("--output-file requires --profile")
        raise SystemExit(2)

    output_path = (
        resolve_workspace_path(
            output,
            workspace.root if workspace is not None else None,
        )
        if output is not None
        else None
    )

    request = build_materialize_run_request(
        project=project,
        profile_name=profile_name,
        overwrite=overwrite,
        artifact_mode=artifact_mode,
        output=output_path,
        command_observability=CommandObservability(
            visuals=visuals,
            heartbeat_interval_seconds=heartbeat_interval_seconds,
            log_level=cli_log_level,
            log_outputs=tuple(cli_log_outputs),
        ),
    )
    if request is None:
        if result_json:
            write_run_results(())
        else:
            logger.info("No enabled materialize profiles; skipping materialize.")
        return
    if result_json:
        validate_result_json_outputs(request)
    configure_profile_logging(cli_log_level, cli_log_outputs)
    results = execute_profile_request(request)
    if result_json:
        write_run_results(results)
