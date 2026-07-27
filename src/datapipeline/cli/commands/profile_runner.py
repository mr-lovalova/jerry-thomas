import argparse
import logging
import time

from datapipeline.cli.output_options import build_cli_output_config
from datapipeline.cli.visuals.execution import route_execution_event
from datapipeline.cli.visuals.rich.progress import visual_summary
from datapipeline.cli.workspace import WorkspaceContext
from datapipeline.execution.events import RunStatus
from datapipeline.execution.observability import CommandFinished
from datapipeline.execution.settings import CommandObservability, LogOutputTarget
from datapipeline.profiles.errors import ProfileCommandError
from datapipeline.profiles.models import BuildRunRequest, ProfileRunRequest
from datapipeline.profiles.orchestration import run_profiles
from datapipeline.profiles.request_builder import (
    build_build_run_request,
    build_runtime_run_request,
)

logger = logging.getLogger(__name__)


def _command_uses_visuals(request: ProfileRunRequest) -> bool:
    if isinstance(request, BuildRunRequest):
        return any(job.settings.observability.visuals == "on" for job in request.jobs)
    return request.artifact_settings.observability.visuals == "on" or any(
        job.observability.visuals == "on" for job in request.jobs
    )


def execute_profile_request(request: ProfileRunRequest) -> None:
    started_at = time.perf_counter()
    status: RunStatus = "error"
    command_error: BaseException | None = None
    try:
        run_profiles(request)
    except ProfileCommandError as exc:
        command_error = exc
        logger.error("%s", exc)
        for note in getattr(exc, "__notes__", ()):
            logger.error("%s", note)
        raise SystemExit(2) from exc
    except (KeyboardInterrupt, SystemExit) as exc:
        command_error = exc
        for note in getattr(exc, "__notes__", ()):
            logger.error("%s", note)
        raise
    except BaseException as exc:
        command_error = exc
        raise
    else:
        status = "success"
    finally:
        try:
            with visual_summary(
                logger.getEffectiveLevel(),
                _command_uses_visuals(request),
            ):
                route_execution_event(
                    CommandFinished(
                        command=request.command,
                        status=status,
                        elapsed_seconds=time.perf_counter() - started_at,
                    ),
                    logger,
                )
        except BaseException as exc:
            if command_error is None:
                raise
            message = f"Reporting command completion also failed: {exc}"
            command_error.add_note(message)
            if isinstance(
                command_error,
                (ProfileCommandError, KeyboardInterrupt, SystemExit),
            ):
                logger.error("%s", message)


def handle_build(
    args: argparse.Namespace,
    cli_log_level: str | None,
    cli_log_outputs: list[LogOutputTarget],
) -> None:
    request = build_build_run_request(
        project=args.project,
        profile_name=args.profile,
        force=args.force,
        command_observability=CommandObservability(
            visuals=args.visuals,
            heartbeat_interval_seconds=args.heartbeat_interval_seconds,
            log_level=cli_log_level,
            log_outputs=tuple(cli_log_outputs),
        ),
    )
    if request is None:
        logger.info("No enabled build profiles; skipping build.")
        return
    execute_profile_request(request)


def handle_serve(
    args: argparse.Namespace,
    workspace: WorkspaceContext | None,
    cli_log_level: str | None,
    cli_log_outputs: list[LogOutputTarget],
) -> None:
    workspace_root = workspace.root if workspace is not None else None
    output = build_cli_output_config(
        transport=args.output_transport,
        fmt=args.output_format,
        directory=args.output_directory,
        output_encoding=args.output_encoding,
        output_compression=args.output_compression,
        workspace_root=workspace_root,
        view=args.output_view,
    )
    request = build_runtime_run_request(
        command="serve",
        project=args.project,
        profile_name=args.profile,
        artifact_mode=args.artifact_mode,
        limit=args.limit,
        preview=args.preview,
        cli_output=output,
        command_observability=CommandObservability(
            visuals=args.visuals,
            heartbeat_interval_seconds=args.heartbeat_interval_seconds,
            log_level=cli_log_level,
            log_outputs=tuple(cli_log_outputs),
        ),
    )
    if request is None:
        logger.info("No enabled serve profiles; skipping serve.")
        return
    execute_profile_request(request)


def handle_inspect(
    args: argparse.Namespace,
    workspace: WorkspaceContext | None,
    cli_log_level: str | None,
    cli_log_outputs: list[LogOutputTarget],
) -> None:
    workspace_root = workspace.root if workspace is not None else None
    output = build_cli_output_config(
        transport=args.output_transport,
        fmt=args.output_format,
        directory=args.output_directory,
        output_encoding=args.output_encoding,
        output_compression=args.output_compression,
        workspace_root=workspace_root,
        view=args.output_view,
    )
    request = build_runtime_run_request(
        command="inspect",
        project=args.project,
        profile_name=args.profile,
        artifact_mode=args.artifact_mode,
        limit=args.limit,
        cli_output=output,
        command_observability=CommandObservability(
            visuals=args.visuals,
            heartbeat_interval_seconds=args.heartbeat_interval_seconds,
            log_level=cli_log_level,
            log_outputs=tuple(cli_log_outputs),
        ),
    )
    if request is None:
        logger.info("No enabled inspect profiles; skipping inspect.")
        return
    execute_profile_request(request)
