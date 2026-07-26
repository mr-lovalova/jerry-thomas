import logging

from datapipeline.cli.commands.profile_runner import execute_profile_request
from datapipeline.cli.workspace import WorkspaceContext
from datapipeline.execution.settings import LogOutputTarget
from datapipeline.profiles.request_builder import build_materialize_run_request
from datapipeline.services.path_policy import resolve_workspace_path

logger = logging.getLogger(__name__)


def handle(
    project: str,
    profile_name: str | None,
    output: str | None,
    overwrite: bool | None,
    artifact_mode: str | None,
    visuals: str | None,
    heartbeat_interval_seconds: float | None,
    cli_log_level: str | None,
    cli_log_outputs: list[LogOutputTarget],
    base_log_level: str,
    workspace: WorkspaceContext | None,
) -> None:
    if profile_name is None and output is not None:
        logger.error("--output requires --profile")
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
        cli_visuals=visuals,
        cli_heartbeat_interval_seconds=heartbeat_interval_seconds,
        cli_log_level=cli_log_level,
        cli_log_outputs=cli_log_outputs,
        base_log_level=base_log_level,
    )
    if request is None:
        logger.info("No enabled materialize profiles; skipping materialize.")
        return
    execute_profile_request(request)
